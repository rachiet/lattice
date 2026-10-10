using System.Text.Json;

namespace Lattice.Tools;

/// The tools a model may call, in the order they were added.
public sealed class ToolRegistry
{
    /// The most tools one registry may hold.
    public const int MaxTools = JsonSchema.MaxProperties;

    readonly List<Tool> _tools = [];

    /// The registered tools, in the order they were added.
    public IReadOnlyList<Tool> Tools => _tools;

    /// Registers a tool.
    ///
    /// `name` must be non-empty printable ASCII without `"` or `\`, because the
    /// name is matched literally as the model writes it. `parameters` must be a
    /// JSON Schema object for the arguments.
    public ToolRegistry Add(string name, string description, string parameters)
    {
        ArgumentNullException.ThrowIfNull(name);
        ArgumentNullException.ThrowIfNull(description);
        ArgumentNullException.ThrowIfNull(parameters);

        if (name.Length == 0 || name.Any(c => c is < ' ' or > '~' or '"' or '\\'))
            throw new ArgumentException(
                $"\"{name}\" is not a valid tool name: use printable ASCII without quotes or backslashes",
                nameof(name));

        if (_tools.Any(t => t.Name == name))
            throw new ArgumentException($"a tool named \"{name}\" is already registered", nameof(name));

        if (_tools.Count == MaxTools)
            throw new InvalidOperationException($"a registry holds at most {MaxTools} tools");

        try
        {
            using var document = JsonDocument.Parse(parameters);
            if (document.RootElement.ValueKind != JsonValueKind.Object)
                throw new ArgumentException("parameters must be a JSON Schema object", nameof(parameters));
        }
        catch (JsonException e)
        {
            throw new ArgumentException($"parameters is not valid JSON: {e.Message}", nameof(parameters), e);
        }

        _tools.Add(new Tool(name, description, parameters));
        return this;
    }
}
