using System.Text.Json;
using Lattice;
using Xunit;

namespace Lattice.Tests;

public class JsonSchemaUnionTests
{
    /// A tool-call union. `get_weather` and `get_time` share `get_`, and in the
    /// `send_email` branch the tag is declared second, so "tag first" cannot be
    /// an accident of declaration order.
    const string Schema =
        """
        {
          "type": "object",
          "oneOf": [
            {
              "properties": {
                "name": { "const": "get_weather" },
                "arguments": {
                  "type": "object",
                  "properties": {
                    "city":  { "type": "string" },
                    "units": { "type": "string", "enum": ["celsius", "fahrenheit"] }
                  },
                  "required": ["city"]
                }
              },
              "required": ["name", "arguments"]
            },
            {
              "properties": {
                "name": { "const": "get_time" },
                "arguments": {
                  "type": "object",
                  "properties": { "zone": { "type": "string" } }
                }
              },
              "required": ["name"]
            },
            {
              "properties": {
                "arguments": {
                  "type": "object",
                  "properties": {
                    "to":   { "type": "string" },
                    "body": { "type": "string" }
                  },
                  "required": ["to", "body"]
                },
                "name": { "const": "send_email" }
              },
              "required": ["name", "arguments"]
            }
          ]
        }
        """;

    static readonly JsonSchema Compiled = JsonSchema.Parse(Schema);

    static (int Accepted, JsonState State) Feed(string json, JsonSchema? schema = null)
    {
        var state = new JsonState(schema ?? Compiled);
        var accepted = 0;
        foreach (var c in json)
        {
            if (!state.TryAdvance(c)) break;
            accepted++;
        }
        return (accepted, state);
    }

    [Theory]
    [InlineData("{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Paris\"}}")]
    [InlineData("{\"name\":\"get_weather\",\"arguments\":{\"units\":\"celsius\",\"city\":\"Paris\"}}")]
    [InlineData("{\"name\":\"get_time\"}")]                              // arguments optional here
    [InlineData("{\"name\":\"get_time\",\"arguments\":{}}")]
    [InlineData("{\"name\":\"send_email\",\"arguments\":{\"to\":\"a\",\"body\":\"b\"}}")]
    [InlineData(" { \"name\" : \"send_email\" , \"arguments\" : { \"body\" : \"b\" , \"to\" : \"a\" } } ")]
    public void CallsMatchingABranchAreAcceptedAndComplete(string json)
    {
        var (accepted, state) = Feed(json);

        Assert.Equal(json.Length, accepted);
        Assert.True(state.IsComplete);
        Assert.True(Parses(json));
    }

    [Theory]
    [InlineData("{\"name\":\"get_")]                                     // get_weather or get_time
    [InlineData("{\"name\":\"get_weather\"")]
    [InlineData("{\"name\":\"get_weather\",\"arguments\":{\"city\":\"P")]
    public void ValidPrefixesSurviveButAreIncomplete(string json)
    {
        var (accepted, state) = Feed(json);

        Assert.Equal(json.Length, accepted);
        Assert.False(state.IsComplete);
    }

    [Theory]
    // --- the tag comes first ------------------------------------------------
    [InlineData("{\"arguments\"", 2)]
    [InlineData("{}", 1)]
    // --- only declared tag values -------------------------------------------
    [InlineData("{\"name\":\"delete_all\"", 9)]
    [InlineData("{\"name\":\"get_x", 13)]
    [InlineData("{\"name\":\"get_\"", 13)]                               // a shared prefix, not a value
    [InlineData("{\"name\":\"get_weather!", 20)]
    [InlineData("{\"name\":1", 8)]
    // --- the chosen branch decides the rest ---------------------------------
    [InlineData("{\"name\":\"get_weather\",\"arguments\":{\"to\"", 36)]   // send_email's argument
    [InlineData("{\"name\":\"send_email\",\"arguments\":{\"city\"", 35)]  // get_weather's argument
    [InlineData("{\"name\":\"get_weather\",\"arguments\":{\"units\":\"kelvin\"", 44)]
    [InlineData("{\"name\":\"get_weather\",\"arguments\":{}", 35)]        // city is required
    [InlineData("{\"name\":\"get_weather\"}", 21)]                        // arguments is required
    [InlineData("{\"name\":\"get_weather\",\"name\"", 23)]               // the tag is written once
    [InlineData("{\"name\":\"get_weather\",\"other\"", 23)]
    public void ViolationsDieAtTheOffendingCharacter(string json, int expectedAccepted)
    {
        var (accepted, _) = Feed(json);

        Assert.Equal(expectedAccepted, accepted);
    }

    [Fact]
    public void AUnionCanBeANestedValue()
    {
        var schema = JsonSchema.Parse(
            """
            {
              "type": "object",
              "properties": {
                "call": {
                  "type": "object",
                  "oneOf": [
                    { "properties": { "kind": { "const": "a" }, "x": { "type": "integer" } } },
                    { "properties": { "kind": { "const": "b" }, "y": { "type": "boolean" } } }
                  ]
                },
                "note": { "type": "string" }
              }
            }
            """);

        var good = "{\"call\":{\"kind\":\"b\",\"y\":true},\"note\":\"n\"}";
        var (accepted, state) = Feed(good, schema);
        Assert.Equal(good.Length, accepted);
        Assert.True(state.IsComplete);

        // After the union closes, the outer object's keys apply again.
        Assert.Equal(21, Feed("{\"call\":{\"kind\":\"b\",\"x\"", schema).Accepted);
    }

    [Fact]
    public void AConstOutsideAUnionIsASingleValueEnum()
    {
        var schema = JsonSchema.Parse(
            "{\"properties\":{\"v\":{\"const\":\"one\"}}}");

        Assert.True(Feed("{\"v\":\"one\"}", schema).State.IsComplete);
        Assert.Equal(8, Feed("{\"v\":\"onx\"}", schema).Accepted);
    }

    [Fact]
    public void AMaskOnlyAllowsRegisteredNames()
    {
        string[] vocab = ["{\"name\":", " \"get_weather\"", " \"delete_all\"", " \"get_", "<eos>"];
        var mask = new JsonMask(vocab, endOfSequence: 4, Compiled);
        var logits = new float[vocab.Length];

        mask.Commit(0);
        mask.Apply(logits);

        Assert.False(float.IsNegativeInfinity(logits[1]));
        Assert.True(float.IsNegativeInfinity(logits[2]));
        Assert.False(float.IsNegativeInfinity(logits[3]));
    }

    /// Every prefix of every call the schema allows is accepted, and the state
    /// finishes exactly at the last character.
    [Fact]
    public void AcceptsEveryPrefixOfEveryMatchingCall()
    {
        string[] calls =
        [
            "{\"name\":\"get_weather\",\"arguments\":{\"city\":\"c\",\"units\":\"fahrenheit\"}}",
            "{\"name\":\"get_time\",\"arguments\":{\"zone\":\"UTC\"}}",
            "{\"name\":\"send_email\",\"arguments\":{\"body\":\"hi\",\"to\":\"x@y\"}}",
        ];

        foreach (var json in calls)
        {
            var state = new JsonState(Compiled);
            for (var i = 0; i < json.Length; i++)
            {
                Assert.True(state.TryAdvance(json[i]), $"rejected index {i} of: {json}");
                Assert.False(state.IsComplete && i < json.Length - 1, $"finished early at {i} of: {json}");
            }
            Assert.True(state.IsComplete);
        }
    }

    // --- compiling ---------------------------------------------------------

    [Theory]
    [InlineData("{\"oneOf\":[]}")]                                                         // empty
    [InlineData("{\"oneOf\":{}}")]                                                         // not an array
    [InlineData("{\"oneOf\":[{\"type\":\"string\"}]}")]                                    // branch is not an object
    [InlineData("{\"oneOf\":[{\"properties\":{\"a\":{\"type\":\"string\"}}}]}")]           // no const property
    [InlineData("{\"properties\":{},\"oneOf\":[{\"properties\":{\"k\":{\"const\":\"a\"}}}]}")]  // both
    [InlineData("{\"oneOf\":[{\"properties\":{\"k\":{\"const\":\"a\"}}},{\"properties\":{\"j\":{\"const\":\"b\"}}}]}")]  // no shared tag
    [InlineData("{\"oneOf\":[{\"properties\":{\"k\":{\"const\":\"a\"}}},{\"properties\":{\"k\":{\"const\":\"a\"}}}]}")]  // duplicate value
    [InlineData("{\"oneOf\":[{\"properties\":{\"k\":{\"const\":\"a\"},\"j\":{\"const\":\"b\"}}}]}")]  // two candidate tags
    public void InvalidUnionsAreRefused(string schema)
    {
        Assert.Throws<ArgumentException>(() => JsonSchema.Parse(schema));
    }

    [Fact]
    public void MoreThanSixtyFourBranchesIsRefused()
    {
        var branches = Enumerable.Range(0, JsonSchema.MaxProperties + 1)
            .Select(i => $"{{\"properties\":{{\"k\":{{\"const\":\"v{i}\"}}}}}}");

        var schema = "{\"oneOf\":[" + string.Join(",", branches) + "]}";

        Assert.Throws<ArgumentException>(() => JsonSchema.Parse(schema));
    }

    static bool Parses(string json)
    {
        try { JsonDocument.Parse(json); return true; }
        catch (JsonException) { return false; }
    }
}
