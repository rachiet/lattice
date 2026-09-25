using System.Text.Json;

namespace Lattice;

/// A type a schema can declare for a value. `Any` is an undeclared type, which
/// accepts any JSON value.
public enum JsonType
{
    Any,
    String,
    Integer,
    Number,
    Boolean,
    Object,
    Array,
    Null,
}

/// A JSON Schema compiled into the form the checker needs: for each object, the
/// declared property names as a prefix table, their types, and which of them are
/// required.
///
/// Instances are immutable and shared — a checker holds a reference and never
/// copies one.
public sealed class JsonSchema
{
    /// Declared objects nested deeper than this are refused when the schema is
    /// compiled.
    public const int MaxDepth = 16;

    /// The most properties one object may declare.
    public const int MaxProperties = 64;

    /// The most schema-governed containers a document may nest. An array and its
    /// item object each take one JSON level, so a schema at MaxDepth reaches
    /// twice that.
    internal const int MaxNesting = MaxDepth * 2;

    internal const int RootNode = 0;

    readonly Node[] _nodes;

    JsonSchema(Node[] nodes) => _nodes = nodes;

    /// Compiles a JSON Schema document.
    ///
    /// The root must declare `"type": "object"` and `"properties"`. Recognised
    /// keywords are `type`, `properties`, `required`, `items` and `enum`; any
    /// other keyword is ignored. Properties the schema does not declare are
    /// never legal, whatever `additionalProperties` says.
    public static JsonSchema Parse(string json)
    {
        ArgumentNullException.ThrowIfNull(json);

        using var document = JsonDocument.Parse(json);
        var root = document.RootElement;

        if (root.ValueKind != JsonValueKind.Object)
            throw new ArgumentException("the schema must be a JSON object", nameof(json));

        if (TypeOf(root) is not (JsonType.Object or JsonType.Any))
            throw new ArgumentException("the root schema must declare \"type\": \"object\"", nameof(json));

        var nodes = new List<Node>();
        if (AddNode(nodes, root, depth: 1) < 0)
            throw new ArgumentException("the root schema must declare \"properties\"", nameof(json));

        return new JsonSchema(nodes.ToArray());
    }

    internal PrefixTable KeysOf(int node) => _nodes[node].Keys;

    internal ulong RequiredOf(int node) => _nodes[node].Required;

    internal Property PropertyAt(int node, int index) => _nodes[node].Properties[index];

    /// Compiles one object schema and returns its node index, or -1 when the
    /// schema declares no properties and the object is therefore free-form.
    static int AddNode(List<Node> nodes, JsonElement schema, int depth)
    {
        if (!schema.TryGetProperty("properties", out var properties)
            || properties.ValueKind != JsonValueKind.Object)
            return -1;

        if (depth > MaxDepth)
            throw new ArgumentException($"objects are nested deeper than {MaxDepth}");

        // Reserved before the children are compiled, because compiling them
        // appends to the same list.
        var index = nodes.Count;
        nodes.Add(null!);

        var names = new List<string>();
        var declared = new List<JsonElement>();
        foreach (var property in properties.EnumerateObject())
        {
            if (names.Contains(property.Name))
                throw new ArgumentException($"\"{property.Name}\" is declared twice");

            names.Add(property.Name);
            declared.Add(property.Value);
        }

        if (names.Count > MaxProperties)
            throw new ArgumentException(
                $"an object declares {names.Count} properties; the limit is {MaxProperties}");

        var compiled = new Property[names.Count];
        for (var i = 0; i < names.Count; i++)
            compiled[i] = BuildProperty(nodes, declared[i], depth);

        nodes[index] = new Node
        {
            Properties = compiled,
            Keys = new PrefixTable(names),
            Required = RequiredMask(schema, names),
        };

        return index;
    }

    static Property BuildProperty(List<Node> nodes, JsonElement schema, int depth)
    {
        if (schema.ValueKind != JsonValueKind.Object)
            return new Property(JsonType.Any, -1, JsonType.Any, -1, null);

        var type = TypeOf(schema);
        var node = -1;
        var itemType = JsonType.Any;
        var itemNode = -1;

        if (type == JsonType.Object)
            node = AddNode(nodes, schema, depth + 1);

        if (type == JsonType.Array
            && schema.TryGetProperty("items", out var items)
            && items.ValueKind == JsonValueKind.Object)
        {
            itemType = TypeOf(items);
            if (itemType == JsonType.Object) itemNode = AddNode(nodes, items, depth + 1);
        }

        var members = StringEnum(schema);
        if (members is not null) type = JsonType.String;

        return new Property(type, node, itemType, itemNode, members);
    }

    /// The `enum` members, when every one of them is a string. An enum with a
    /// non-string member is ignored.
    static PrefixTable? StringEnum(JsonElement schema)
    {
        if (!schema.TryGetProperty("enum", out var choices)
            || choices.ValueKind != JsonValueKind.Array
            || choices.GetArrayLength() == 0)
            return null;

        var members = new List<string>();
        foreach (var choice in choices.EnumerateArray())
        {
            if (choice.ValueKind != JsonValueKind.String) return null;
            members.Add(choice.GetString()!);
        }

        if (members.Count > MaxProperties)
            throw new ArgumentException(
                $"an enum has {members.Count} members; the limit is {MaxProperties}");

        return new PrefixTable(members);
    }

    static ulong RequiredMask(JsonElement schema, List<string> names)
    {
        if (!schema.TryGetProperty("required", out var required)
            || required.ValueKind != JsonValueKind.Array)
            return 0;

        var mask = 0UL;
        foreach (var name in required.EnumerateArray())
        {
            var index = names.IndexOf(name.GetString() ?? "");
            if (index < 0)
                throw new ArgumentException(
                    $"\"required\" names {name}, which is not a declared property");

            mask |= 1UL << index;
        }

        return mask;
    }

    static JsonType TypeOf(JsonElement schema) =>
        schema.TryGetProperty("type", out var type) && type.ValueKind == JsonValueKind.String
            ? type.GetString() switch
            {
                "string" => JsonType.String,
                "integer" => JsonType.Integer,
                "number" => JsonType.Number,
                "boolean" => JsonType.Boolean,
                "object" => JsonType.Object,
                "array" => JsonType.Array,
                "null" => JsonType.Null,
                var other => throw new ArgumentException($"unknown type \"{other}\""),
            }
            : JsonType.Any;

    sealed class Node
    {
        public required Property[] Properties { get; init; }
        public required PrefixTable Keys { get; init; }
        public required ulong Required { get; init; }
    }
}

/// One declared property's value constraints.
///
/// `Node` is the object schema the value must match, or -1 when it is free-form.
/// `ItemType` and `ItemNode` describe an array's items.
readonly record struct Property(
    JsonType Type,
    int Node,
    JsonType ItemType,
    int ItemNode,
    PrefixTable? Enum);

/// Answers "which of these strings have character `c` at position `p`" in one
/// lookup, as a bitmask with one bit per string.
///
/// A set of candidate strings is a mask; narrowing it by a character is an `and`
/// with the cell for that position and character.
sealed class PrefixTable
{
    const int Alphabet = 128;

    readonly ulong[] _cells;    // position * Alphabet + character
    readonly ulong[] _endings;  // position -> strings whose length is exactly that

    public PrefixTable(IReadOnlyList<string> patterns)
    {
        All = patterns.Count == 64 ? ulong.MaxValue : (1UL << patterns.Count) - 1;

        var longest = 0;
        foreach (var pattern in patterns) longest = Math.Max(longest, pattern.Length);

        _cells = new ulong[longest * Alphabet];
        _endings = new ulong[longest + 1];

        for (var k = 0; k < patterns.Count; k++)
        {
            var pattern = patterns[k];
            for (var p = 0; p < pattern.Length; p++)
            {
                if (pattern[p] >= Alphabet)
                    throw new ArgumentException($"\"{pattern}\" is not ASCII");

                _cells[p * Alphabet + pattern[p]] |= 1UL << k;
            }

            _endings[pattern.Length] |= 1UL << k;
        }
    }

    /// Every string in the table.
    public ulong All { get; }

    /// The strings with `c` at `position`.
    public ulong At(int position, char c) =>
        position * Alphabet + c < _cells.Length && c < Alphabet
            ? _cells[position * Alphabet + c]
            : 0;

    /// The strings whose length is exactly `position`, so a string that ends
    /// here is complete rather than a prefix of a longer one.
    public ulong EndingAt(int position) =>
        position < _endings.Length ? _endings[position] : 0;
}
