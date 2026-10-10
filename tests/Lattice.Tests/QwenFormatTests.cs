using Lattice.Tools;
using Lattice.Tools.Formats;
using Xunit;

namespace Lattice.Tests;

public class QwenFormatTests
{
    static readonly QwenFormat Format = new();

    static ToolRegistry Tools() => new ToolRegistry()
        .Add("get_weather", "Get the current weather for a city.",
            """
            {
              "type": "object",
              "properties": {
                "city":  { "type": "string" },
                "units": { "type": "string", "enum": ["celsius", "fahrenheit"] }
              },
              "required": ["city"]
            }
            """)
        .Add("send_email", "Send an email.",
            """{"type":"object","properties":{"to":{"type":"string"},"body":{"type":"string"}},"required":["to","body"]}""");

    // --- prompt ------------------------------------------------------------

    [Fact]
    public void ThePromptMatchesTheChatTemplate()
    {
        var prompt = Format.Prompt(Tools(), "What's the weather in Paris?");

        var expected =
            "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n\n" +
            "# Tools\n\nYou may call one or more functions to assist with the user query.\n\n" +
            "You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n" +
            "{\"type\": \"function\", \"function\": {\"name\": \"get_weather\", \"description\": \"Get the current weather for a city.\", " +
            "\"parameters\": {\"type\": \"object\", \"properties\": {\"city\": {\"type\": \"string\"}, " +
            "\"units\": {\"type\": \"string\", \"enum\": [\"celsius\", \"fahrenheit\"]}}, \"required\": [\"city\"]}}}\n" +
            "{\"type\": \"function\", \"function\": {\"name\": \"send_email\", \"description\": \"Send an email.\", " +
            "\"parameters\": {\"type\": \"object\", \"properties\": {\"to\": {\"type\": \"string\"}, " +
            "\"body\": {\"type\": \"string\"}}, \"required\": [\"to\", \"body\"]}}}\n" +
            "</tools>\n\nFor each function call, return a json object with function name and arguments " +
            "within <tool_call></tool_call> XML tags:\n<tool_call>\n" +
            "{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n" +
            "<|im_start|>user\nWhat's the weather in Paris?<|im_end|>\n" +
            "<|im_start|>assistant\n<tool_call>\n";

        Assert.Equal(expected, prompt);
    }

    [Fact]
    public void ACustomSystemMessageReplacesTheDefault()
    {
        var prompt = Format.Prompt(Tools(), "hi", system: "Be brief.");

        Assert.StartsWith("<|im_start|>system\nBe brief.\n\n# Tools", prompt);
    }

    [Fact]
    public void StringsAreEscapedAndNonAsciiIsKept()
    {
        var tools = new ToolRegistry().Add("t", "Says \"hé\"\nthen\\stops", "{}");

        Assert.Contains("\"description\": \"Says \\\"hé\\\"\\nthen\\\\stops\"", Format.Prompt(tools, "x"));
    }

    [Fact]
    public void AnEmptyRegistryIsRefused()
    {
        Assert.Throws<ArgumentException>(() => Format.Prompt(new ToolRegistry(), "x"));
        Assert.Throws<ArgumentException>(() => Format.Schema(new ToolRegistry()));
    }

    // --- schema ------------------------------------------------------------

    [Theory]
    [InlineData("{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}")]
    [InlineData("{\"name\": \"get_weather\", \"arguments\": {\"units\": \"celsius\", \"city\": \"Paris\"}}")]
    [InlineData("{\"name\": \"send_email\", \"arguments\": {\"to\": \"a@b\", \"body\": \"hi\"}}")]
    public void TheSchemaAcceptsCallsToRegisteredTools(string call)
    {
        var state = new JsonState(JsonSchema.Parse(Format.Schema(Tools())));

        foreach (var c in call)
            Assert.True(state.TryAdvance(c), $"rejected {call}");
        Assert.True(state.IsComplete);
    }

    [Theory]
    [InlineData("{\"name\": \"delete_all\"")]                                           // not registered
    [InlineData("{\"arguments\"")]                                                      // name comes first
    [InlineData("{\"name\": \"get_weather\"}")]                                         // arguments required
    [InlineData("{\"name\": \"get_weather\", \"arguments\": {\"to\"")]                  // another tool's argument
    [InlineData("{\"name\": \"get_weather\", \"arguments\": {\"units\": \"kelvin\"")]   // not in the enum
    [InlineData("{\"name\": \"send_email\", \"arguments\": {\"to\": \"a\"}")]           // body required
    public void TheSchemaRefusesAnythingElse(string call)
    {
        var state = new JsonState(JsonSchema.Parse(Format.Schema(Tools())));

        var accepted = true;
        foreach (var c in call)
            if (!state.TryAdvance(c)) { accepted = false; break; }

        Assert.False(accepted && state.IsComplete);
    }

    // --- parse -------------------------------------------------------------

    [Theory]
    [InlineData("{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}")]
    [InlineData("{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>")]
    [InlineData("<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>\ntrailing")]
    [InlineData("  {\"name\":\"get_weather\",\"arguments\":{\"city\":\"Paris\"}}  ")]
    public void ACallIsReadFromTheOutput(string output)
    {
        var call = Format.Parse(output);

        Assert.NotNull(call);
        Assert.Equal("get_weather", call.Name);
        Assert.Equal("Paris", System.Text.Json.JsonDocument.Parse(call.Arguments).RootElement.GetProperty("city").GetString());
    }

    [Theory]
    [InlineData("")]
    [InlineData("The weather is sunny.")]
    [InlineData("{\"name\": \"get_weather\"")]                                     // cut off
    [InlineData("{\"name\": \"get_weather\"}")]                                    // no arguments
    [InlineData("{\"name\": 1, \"arguments\": {}}")]                               // name is not a string
    [InlineData("{\"name\": \"get_weather\", \"arguments\": \"Paris\"}")]          // arguments is not an object
    [InlineData("[{\"name\": \"get_weather\", \"arguments\": {}}]")]               // not an object
    public void MalformedOutputIsNotACall(string output)
    {
        Assert.Null(Format.Parse(output));
    }
}
