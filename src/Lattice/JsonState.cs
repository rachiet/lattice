using System.Diagnostics.CodeAnalysis;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace Lattice;

/// What the next character is allowed to be, given everything read so far.
///
/// Inside a number, a `*Digit` name owes a character and cannot end the number;
/// a `More*` name may either continue or end it.
public enum Expect
{
    /// Only `{`. The document must be a JSON object, so `123`, `[1]` and `"hi"`
    /// are refused at the first character. The zero value, so a new JsonState
    /// starts here.
    Object,

    /// A value must start here: after `:`, or after `,` in an array.
    Value,
    /// A value, or `]` to close an empty array. Reached from `[`.
    ValueOrClose,
    /// A key's opening quote, or `}` to close an empty object. Reached from `{`.
    KeyOrClose,
    /// A key's opening quote. No `}` — `{"a":1,}` is invalid. Reached from a
    /// `,` inside an object.
    Key,
    /// More of the key, or the `"` that closes it.
    KeyText,
    /// Only `:`. Reached from a key's closing quote.
    Colon,
    /// More of the string, or the `"` that closes it.
    StringText,
    /// More of a string value the schema restricts to an `enum`, or the `"` that
    /// closes it.
    EnumText,
    /// A `,`, or the closer matching the innermost container. Reached whenever
    /// a value finishes.
    CommaOrClose,
    /// Only a digit. Reached from `-`.
    FirstDigit,
    /// `.`, `e`/`E`, or the number ends here. No further digit — that would be
    /// a leading zero. Reached from a `0`.
    FractionOrExponent,
    /// More digits, `.`, `e`/`E`, or the number ends here.
    MoreInteger,
    /// Only a digit. Reached from `.`.
    FractionDigit,
    /// More digits, `e`/`E`, or the number ends here.
    MoreFraction,
    /// `+`, `-`, or a digit. Reached from `e`/`E`.
    ExponentSignOrDigit,
    /// Only a digit. Reached from `+` or `-`.
    ExponentDigit,
    /// More digits, or the number ends here.
    MoreExponent,
    /// The next letter of `true`, `false` or `null`, and nothing else.
    LiteralLetter,
    /// Only whitespace. The object closed and depth is back to 0.
    End,
}

/// A character-level JSON checker: does this character keep the document a
/// valid JSON prefix, and is that prefix now complete?
///
/// Copying a JsonState copies the whole state: every field is a value type
/// except the schema, which is immutable and shared. Open containers are packed
/// into a ulong bitmask, one bit each.
///
/// Without a schema the checker accepts any JSON object. With one it also
/// enforces the declared property names, their types, the required ones, any
/// enums, and tagged unions — the tag first, then the branch it selects.
public struct JsonState
{
    /// Objects and arrays nested deeper than this are rejected.
    public const int MaxDepth = 64;

    Expect _expect;
    ulong _containers;  // one bit per open container, innermost in bit (_depth - 1): 1 = array, 0 = object
    int _depth;
    bool _escaped;      // the previous character was a backslash inside a string
    int _hexLeft;       // remaining hex digits owed to a \u escape
    byte _literal;      // which word is being matched: 1 = true, 2 = false, 3 = null
    byte _literalIndex; // how much of that word has been consumed

    readonly JsonSchema? _schema;  // null: any-shape JSON, no schema checks
    FrameStack _frames;  // one per open container the schema governs, indexed by depth - 1
    int _freeDepth;      // containers open inside a free-form value
    ulong _candidates;   // names still possible for the key or enum member being read
    int _textPos;        // characters of that name read so far
    int _property;       // the property the last closed key resolved to
    bool _integerOnly;   // the number being read may not have a fraction or exponent

    /// A checker that accepts any JSON object.
    public JsonState()
    {
    }

    /// A checker that accepts only JSON objects matching `schema`.
    public JsonState(JsonSchema schema)
    {
        ArgumentNullException.ThrowIfNull(schema);
        _schema = schema;
    }

    public readonly Expect Expecting => _expect;
    public readonly int Depth => _depth;

    /// True once the object has closed — the `}` that takes depth back to 0.
    public readonly bool IsComplete => _expect == Expect.End;

    /// Feeds one character. Returns true if the document is still valid and the
    /// state has advanced; false if the character breaks it, in which case the
    /// state may have partly advanced and must be discarded.
    public bool TryAdvance(char c) => _expect switch
    {
        Expect.KeyText or Expect.StringText => AdvanceString(c),

        Expect.EnumText => AdvanceEnum(c),

        Expect.Object => IsSpace(c) || c == '{'
            && Push(array: false, Expect.KeyOrClose, _schema is not null, JsonSchema.RootNode),

        Expect.Value => IsSpace(c) || BeginValue(c),

        Expect.ValueOrClose => IsSpace(c) || c switch
        {
            ']' => Pop(array: true),
            _ => BeginValue(c),
        },

        Expect.KeyOrClose => IsSpace(c) || c switch
        {
            '"' => BeginKey(),
            '}' => Pop(array: false),
            _ => false,
        },

        Expect.Key => IsSpace(c) || c == '"' && BeginKey(),

        Expect.Colon => IsSpace(c) || c == ':' && Enter(Expect.Value),

        Expect.CommaOrClose => AdvanceCommaOrClose(c),

        // A number has no closing character: it ends when a delimiter arrives,
        // which is then re-read from CommaOrClose.
        Expect.FirstDigit => c == '0' ? Enter(Expect.FractionOrExponent)
                           : IsDigit19(c) && Enter(Expect.MoreInteger),

        Expect.FractionOrExponent => c switch
        {
            '.' when !_integerOnly => Enter(Expect.FractionDigit),
            'e' or 'E' when !_integerOnly => Enter(Expect.ExponentSignOrDigit),
            _ => EndNumber(c),  // a second digit would be a leading zero
        },

        Expect.MoreInteger => IsDigit(c) || c switch
        {
            '.' when !_integerOnly => Enter(Expect.FractionDigit),
            'e' or 'E' when !_integerOnly => Enter(Expect.ExponentSignOrDigit),
            _ => EndNumber(c),
        },

        Expect.FractionDigit => IsDigit(c) && Enter(Expect.MoreFraction),

        Expect.MoreFraction => IsDigit(c) || c switch
        {
            'e' or 'E' => Enter(Expect.ExponentSignOrDigit),
            _ => EndNumber(c),
        },

        Expect.ExponentSignOrDigit => c is '+' or '-' ? Enter(Expect.ExponentDigit)
                         : IsDigit(c) && Enter(Expect.MoreExponent),

        Expect.ExponentDigit => IsDigit(c) && Enter(Expect.MoreExponent),

        Expect.MoreExponent => IsDigit(c) || EndNumber(c),

        Expect.LiteralLetter => AdvanceLiteral(c),

        Expect.End => IsSpace(c),

        _ => false,
    };

    bool AdvanceCommaOrClose(char c)
    {
        if (IsSpace(c)) return true;
        return c switch
        {
            ',' when _depth > 0 => Enter(InnermostIsArray() ? Expect.Value : Expect.Key),
            '}' => Pop(array: false),
            ']' => Pop(array: true),
            _ => false,
        };
    }

    bool AdvanceString(char c)
    {
        // A declared name is matched literally against the schema's prefix table,
        // so none of the escape machinery below applies to one.
        if (_expect == Expect.KeyText && SchemaActive)
        {
            if (c == '"') return CloseKey();
            if (c == '\\') return false;
            return Narrow(_schema!.KeysOf(CurrentFrame.Node), c);
        }

        // \u must be followed by exactly four hex digits before anything else.
        if (_hexLeft > 0)
        {
            if (!IsHex(c)) return false;
            _hexLeft--;
            return true;
        }

        if (_escaped)
        {
            if (c == 'u') _hexLeft = 4;
            else if (c is not ('"' or '\\' or '/' or 'b' or 'f' or 'n' or 'r' or 't')) return false;
            _escaped = false;
            return true;
        }

        if (c == '\\') { _escaped = true; return true; }

        // A quote closes the key or the string value.
        if (c == '"') return _expect == Expect.KeyText ? Enter(Expect.Colon) : EndValue();

        return c >= 0x20;  // raw control characters must be escaped
    }

    bool AdvanceLiteral(char c)
    {
        var word = _literal switch { 1 => "true", 2 => "false", _ => "null" };
        if (c != word[_literalIndex]) return false;
        _literalIndex++;
        return _literalIndex == word.Length ? EndValue() : true;
    }

    bool BeginValue(char c)
    {
        var declared = JsonType.Any;
        var node = -1;
        var itemType = JsonType.Any;
        var itemNode = -1;
        PrefixTable? members = null;

        if (SchemaActive)
        {
            if (InnermostIsArray())
            {
                // Every item of an array shares one declared type.
                declared = CurrentFrame.ItemType;
                node = CurrentFrame.Node;
            }
            else
            {
                var property = _schema!.PropertyAt(CurrentFrame.Node, _property);
                declared = property.Type;
                node = property.Node;
                itemType = property.ItemType;
                itemNode = property.ItemNode;
                members = property.Enum;
            }
        }

        switch (c)
        {
            case '{':
                return Allows(declared, JsonType.Object)
                    && Push(array: false, Expect.KeyOrClose, node >= 0, node);
            case '[':
                return Allows(declared, JsonType.Array)
                    && Push(array: true, Expect.ValueOrClose, SchemaActive, itemNode, itemType);
            case '"':
                if (!Allows(declared, JsonType.String)) return false;
                return members is null ? Enter(Expect.StringText) : BeginEnum(members);
            case '-':
                if (!AllowsNumber(declared)) return false;
                _integerOnly = declared is JsonType.Integer;
                return Enter(Expect.FirstDigit);
            case '0':
                if (!AllowsNumber(declared)) return false;
                _integerOnly = declared is JsonType.Integer;
                return Enter(Expect.FractionOrExponent);
            case 't': return Allows(declared, JsonType.Boolean) && BeginLiteral(1);
            case 'f': return Allows(declared, JsonType.Boolean) && BeginLiteral(2);
            case 'n': return Allows(declared, JsonType.Null) && BeginLiteral(3);
            default:
                if (!IsDigit19(c) || !AllowsNumber(declared)) return false;
                _integerOnly = declared is JsonType.Integer;
                return Enter(Expect.MoreInteger);
        }
    }

    bool BeginLiteral(byte literal)
    {
        _literal = literal;
        _literalIndex = 1;  // the first letter is the character we just consumed
        _expect = Expect.LiteralLetter;
        return true;
    }

    /// Opens a container. `governed` says whether the schema constrains what goes
    /// inside it; `node` is the object schema for an object, or an array's item
    /// schema, and `itemType` the type of an array's items.
    bool Push(bool array, Expect next, bool governed = false, int node = -1,
              JsonType itemType = JsonType.Any)
    {
        if (_depth == MaxDepth) return false;
        if (array) _containers |= 1UL << _depth;
        else _containers &= ~(1UL << _depth);
        _depth++;
        _expect = next;

        if (_schema is null) return true;

        // A free-form value has no frame, and every container inside it is
        // free-form too, so governed levels are always the outermost ones and
        // depth - 1 stays their index.
        if (_freeDepth > 0 || !governed)
        {
            _freeDepth++;
            return true;
        }

        if (_depth > JsonSchema.MaxNesting) return false;

        CurrentFrame = new Frame { Node = node, ItemType = itemType };
        return true;
    }

    bool Pop(bool array)
    {
        if (_depth == 0 || InnermostIsArray() != array) return false;

        // An object cannot close while a required property is still unwritten.
        if (SchemaActive && !array
            && (_schema!.RequiredOf(CurrentFrame.Node) & ~CurrentFrame.Used) != 0)
            return false;

        if (_schema is not null && _freeDepth > 0) _freeDepth--;

        _depth--;
        return EndValue();
    }

    /// A value just completed. At depth 0 that was the whole document.
    bool EndValue()
    {
        _integerOnly = false;
        _expect = _depth == 0 ? Expect.End : Expect.CommaOrClose;
        return true;
    }

    /// Close the number, then let CommaOrClose (or End) rule on the delimiter.
    /// Terminates: EndValue never lands back on a number position.
    bool EndNumber(char c)
    {
        EndValue();
        return TryAdvance(c);
    }

    bool Enter(Expect next)
    {
        _expect = next;
        return true;
    }

    /// True when the schema constrains the position being read: there is a schema
    /// and the position is not inside a free-form value.
    readonly bool SchemaActive => _schema is not null && _freeDepth == 0;

    /// The frame for the innermost governed container.
    [UnscopedRef]
    ref Frame CurrentFrame => ref _frames[_depth - 1];

    /// A key's opening quote. Every declared name this object has not written yet
    /// is a candidate.
    bool BeginKey()
    {
        _expect = Expect.KeyText;
        if (!SchemaActive) return true;

        _candidates = _schema!.KeysOf(CurrentFrame.Node).All & ~CurrentFrame.Used;
        _textPos = 0;
        return _candidates != 0;  // every declared name is already written
    }

    /// Drops every candidate that does not have `c` at the current position.
    bool Narrow(PrefixTable table, char c)
    {
        _candidates &= table.At(_textPos, c);
        _textPos++;
        return _candidates != 0;
    }

    /// A key's closing quote. One candidate ends here, and it fixes the type of
    /// the value that follows.
    bool CloseKey()
    {
        var finished = _candidates & _schema!.KeysOf(CurrentFrame.Node).EndingAt(_textPos);
        if (finished == 0) return false;  // a prefix of a declared name, not one

        _property = BitOperations.TrailingZeroCount(finished);
        CurrentFrame.Used |= 1UL << _property;
        return Enter(Expect.Colon);
    }

    bool BeginEnum(PrefixTable members)
    {
        _candidates = members.All;
        _textPos = 0;
        return Enter(Expect.EnumText);
    }

    bool AdvanceEnum(char c)
    {
        var members = _schema!.PropertyAt(CurrentFrame.Node, _property).Enum!;

        if (c == '"')
        {
            var finished = _candidates & members.EndingAt(_textPos);
            if (finished == 0) return false;

            // A union's tag picks the branch the rest of this object follows. The
            // tag is already written, so it counts as used in the branch.
            if (_schema!.TryGetBranch(CurrentFrame.Node, BitOperations.TrailingZeroCount(finished),
                                     out var branch, out var tag))
            {
                CurrentFrame.Node = branch;
                CurrentFrame.Used = 1UL << tag;
            }

            return EndValue();
        }

        if (c == '\\') return false;  // members are matched literally
        return Narrow(members, c);
    }

    static bool Allows(JsonType declared, JsonType actual) =>
        declared is JsonType.Any || declared == actual;

    static bool AllowsNumber(JsonType declared) =>
        declared is JsonType.Any or JsonType.Number or JsonType.Integer;

    readonly bool InnermostIsArray() => (_containers & 1UL << _depth - 1) != 0;

    static bool IsSpace(char c) => c is ' ' or '\t' or '\n' or '\r';
    static bool IsDigit(char c) => c is >= '0' and <= '9';
    static bool IsDigit19(char c) => c is >= '1' and <= '9';
    static bool IsHex(char c) => IsDigit(c) || c is >= 'a' and <= 'f' or >= 'A' and <= 'F';
}

/// What one open container needs from the schema.
struct Frame
{
    /// Declared properties already written in this object.
    public ulong Used;

    /// The object schema for this level, or an array's item schema. -1 when the
    /// level holds no object schema.
    public int Node;

    /// The declared type of an array's items.
    public JsonType ItemType;
}

/// The frames of the open containers, innermost at `Depth - 1`.
[InlineArray(JsonSchema.MaxNesting)]
struct FrameStack
{
    Frame _first;
}
