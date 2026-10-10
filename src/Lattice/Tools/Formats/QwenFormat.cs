using System.Text;
using System.Text.Json;

namespace Lattice.Tools.Formats;

/// The tool-calling format of Qwen2.5 instruct models.
///
/// Tools are listed inside `<tools></tools>` in the system turn, one JSON
/// object per line, as the model's own chat template renders them. The model
/// answers with `<tool_call>{"name": ..., "arguments": ...}</tool_call>`.
public sealed class QwenFormat : IToolFormat
{
    /// The system message used when none is given.
    public const string DefaultSystem =
        "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.";

    const string CallOpen = "<tool_call>";
    const string CallClose = "</tool_call>";

    /// Ends with the assistant turn open and `<tool_call>` already written, so
    /// the model's reply is the call's JSON.
    public string Prompt(ToolRegistry tools, string request, string? system = null)
    {
        ArgumentNullException.ThrowIfNull(tools);
        ArgumentNullException.ThrowIfNull(request);
        RequireTools(tools);

        var prompt = new StringBuilder();
        prompt.Append("<|im_start|>system\n").Append(system ?? DefaultSystem);
        prompt.Append("\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n");
        prompt.Append("You are provided with function signatures within <tools></tools> XML tags:\n<tools>");

        foreach (var tool in tools.Tools)
        {
            prompt.Append('\n');
            prompt.Append("{\"type\": \"function\", \"function\": {\"name\": ");
            WriteString(prompt, tool.Name);
            prompt.Append(", \"description\": ");
            WriteString(prompt, tool.Description);
            prompt.Append(", \"parameters\": ");
            using (var parameters = JsonDocument.Parse(tool.Parameters))
                WriteValue(prompt, parameters.RootElement);
            prompt.Append("}}");
        }

        // The template's own text, double braces included.
        prompt.Append("\n</tools>\n\nFor each function call, return a json object with function name and ");
        prompt.Append("arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n");
        prompt.Append("{{\"name\": <function-name>, \"arguments\": <args-json-object>}}\n</tool_call><|im_end|>\n");

        prompt.Append("<|im_start|>user\n").Append(request).Append("<|im_end|>\n");
        prompt.Append("<|im_start|>assistant\n").Append(CallOpen).Append('\n');
        return prompt.ToString();
    }

    /// A `oneOf` with one branch per tool. Each branch fixes `name` to the
    /// tool's name and gives `arguments` the tool's parameter schema; both are
    /// required.
    public string Schema(ToolRegistry tools)
    {
        ArgumentNullException.ThrowIfNull(tools);
        RequireTools(tools);

        var schema = new StringBuilder("{\"type\":\"object\",\"oneOf\":[");
        for (var i = 0; i < tools.Tools.Count; i++)
        {
            var tool = tools.Tools[i];
            if (i > 0) schema.Append(',');
            schema.Append("{\"properties\":{\"name\":{\"const\":");
            WriteString(schema, tool.Name);
            schema.Append("},\"arguments\":").Append(tool.Parameters);
            schema.Append("},\"required\":[\"name\",\"arguments\"]}");
        }

        return schema.Append("]}").ToString();
    }

    /// Accepts the bare JSON or the JSON followed by `</tool_call>`, and ignores
    /// anything after that tag. A leading `<tool_call>` is also accepted.
    public ToolCall? Parse(string output)
    {
        ArgumentNullException.ThrowIfNull(output);

        var text = output.Trim();
        if (text.StartsWith(CallOpen, StringComparison.Ordinal)) text = text[CallOpen.Length..];

        var close = text.IndexOf(CallClose, StringComparison.Ordinal);
        if (close >= 0) text = text[..close];

        try
        {
            using var document = JsonDocument.Parse(text);
            var root = document.RootElement;

            if (root.ValueKind != JsonValueKind.Object
                || !root.TryGetProperty("name", out var name) || name.ValueKind != JsonValueKind.String
                || !root.TryGetProperty("arguments", out var arguments) || arguments.ValueKind != JsonValueKind.Object)
                return null;

            return new ToolCall(name.GetString()!, arguments.GetRawText());
        }
        catch (JsonException)
        {
            return null;
        }
    }

    static void RequireTools(ToolRegistry tools)
    {
        if (tools.Tools.Count == 0)
            throw new ArgumentException("the registry has no tools", nameof(tools));
    }

    /// Writes JSON with `", "` and `": "` between items and non-ASCII left
    /// as is, the way the chat template serialises tools.
    static void WriteValue(StringBuilder text, JsonElement value)
    {
        switch (value.ValueKind)
        {
            case JsonValueKind.Object:
                text.Append('{');
                var firstProperty = true;
                foreach (var property in value.EnumerateObject())
                {
                    if (!firstProperty) text.Append(", ");
                    firstProperty = false;
                    WriteString(text, property.Name);
                    text.Append(": ");
                    WriteValue(text, property.Value);
                }
                text.Append('}');
                break;

            case JsonValueKind.Array:
                text.Append('[');
                var firstItem = true;
                foreach (var item in value.EnumerateArray())
                {
                    if (!firstItem) text.Append(", ");
                    firstItem = false;
                    WriteValue(text, item);
                }
                text.Append(']');
                break;

            case JsonValueKind.String:
                WriteString(text, value.GetString()!);
                break;

            default:
                text.Append(value.GetRawText());  // numbers, true, false, null
                break;
        }
    }

    static void WriteString(StringBuilder text, string value)
    {
        text.Append('"');
        foreach (var c in value)
        {
            switch (c)
            {
                case '"': text.Append("\\\""); break;
                case '\\': text.Append("\\\\"); break;
                case '\n': text.Append("\\n"); break;
                case '\r': text.Append("\\r"); break;
                case '\t': text.Append("\\t"); break;
                case '\b': text.Append("\\b"); break;
                case '\f': text.Append("\\f"); break;
                default:
                    if (c < ' ') text.Append($"\\u{(int)c:x4}");
                    else text.Append(c);
                    break;
            }
        }
        text.Append('"');
    }
}
