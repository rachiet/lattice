namespace Lattice.Tools;

/// How one model family is prompted with tools and how it writes a call.
public interface IToolFormat
{
    /// The prompt for one request, describing every registered tool. It ends
    /// where the model starts writing the call's JSON.
    string Prompt(ToolRegistry tools, string request, string? system = null);

    /// A JSON Schema matching exactly one call to any registered tool, ready for
    /// `JsonSchema.Parse`.
    string Schema(ToolRegistry tools);

    /// Reads the call from the text the model generated after the prompt, or
    /// returns null when the text holds no well-formed call. Does not check the
    /// arguments against the tool's schema.
    ToolCall? Parse(string output);
}
