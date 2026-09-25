using System.Text.Json;
using Lattice;
using Xunit;

namespace Lattice.Tests;

public class JsonSchemaTests
{
    /// Property names chosen so the prefix matching has work to do: `name` and
    /// `nickname` share `n`, `age`, `active` and `address` share `a`, and the
    /// enum members `admin` and `adult` share `ad`.
    const string Schema =
        """
        {
          "type": "object",
          "properties": {
            "name":     { "type": "string" },
            "nickname": { "type": "string" },
            "age":      { "type": "integer" },
            "score":    { "type": "number" },
            "active":   { "type": "boolean" },
            "role":     { "type": "string", "enum": ["admin", "adult", "user"] },
            "tags":     { "type": "array", "items": { "type": "string" } },
            "address":  {
              "type": "object",
              "properties": { "city": { "type": "string" }, "zip": { "type": "integer" } },
              "required": ["city"]
            },
            "meta":     { "type": "object" }
          },
          "required": ["name"]
        }
        """;

    static readonly JsonSchema Compiled = JsonSchema.Parse(Schema);

    /// Feeds a string character by character. Returns how many characters were
    /// accepted before one was rejected, and the state it ended in.
    static (int Accepted, JsonState State) Feed(string json)
    {
        var state = new JsonState(Compiled);
        var accepted = 0;
        foreach (var c in json)
        {
            if (!state.TryAdvance(c)) break;
            accepted++;
        }
        return (accepted, state);
    }

    [Theory]
    // --- documents the schema allows --------------------------------------
    [InlineData("{\"name\":\"x\"}")]
    [InlineData("{\"name\":\"\"}")]
    [InlineData("{\"name\":\"x\",\"nickname\":\"y\"}")]
    [InlineData("{\"name\":\"x\",\"age\":42}")]
    [InlineData("{\"name\":\"x\",\"age\":-7}")]
    [InlineData("{\"name\":\"x\",\"age\":0}")]
    [InlineData("{\"name\":\"x\",\"score\":1.5}")]
    [InlineData("{\"name\":\"x\",\"score\":-0.5e-10}")]
    [InlineData("{\"name\":\"x\",\"score\":3}")]          // an integer is a number
    [InlineData("{\"name\":\"x\",\"active\":true}")]
    [InlineData("{\"name\":\"x\",\"active\":false}")]
    [InlineData("{\"name\":\"x\",\"role\":\"admin\"}")]
    [InlineData("{\"name\":\"x\",\"role\":\"adult\"}")]
    [InlineData("{\"name\":\"x\",\"role\":\"user\"}")]
    [InlineData("{\"name\":\"x\",\"tags\":[]}")]
    [InlineData("{\"name\":\"x\",\"tags\":[\"a\",\"b\"]}")]
    [InlineData("{\"name\":\"x\",\"address\":{\"city\":\"y\"}}")]
    [InlineData("{\"name\":\"x\",\"address\":{\"city\":\"y\",\"zip\":12}}")]
    [InlineData("{\"name\":\"x\",\"address\":{\"zip\":12,\"city\":\"y\"}}")]
    [InlineData("{\"name\":\"x\",\"meta\":{}}")]
    [InlineData("{\"name\":\"x\",\"meta\":{\"anything\":[1,{\"deep\":null}]}}")]
    [InlineData("  {  \"name\"  :  \"x\"  }  ")]
    [InlineData("{\"name\":\"esc\\n\\u00e9\"}")]          // string values keep every escape
    public void MatchingDocumentsAreAcceptedAndComplete(string json)
    {
        var (accepted, state) = Feed(json);

        Assert.Equal(json.Length, accepted);
        Assert.True(state.IsComplete);
        Assert.True(Parses(json));
    }

    [Theory]
    // --- valid prefixes: still alive, not yet a document -------------------
    [InlineData("{")]
    [InlineData("{\"")]
    [InlineData("{\"n")]                                   // name or nickname
    [InlineData("{\"ni")]                                  // nickname only
    [InlineData("{\"a")]                                   // age, active or address
    [InlineData("{\"name\"")]
    [InlineData("{\"name\":")]
    [InlineData("{\"name\":\"x")]
    [InlineData("{\"name\":\"x\",")]
    [InlineData("{\"name\":\"x\",\"ag")]
    [InlineData("{\"name\":\"x\",\"age\":4")]
    [InlineData("{\"name\":\"x\",\"role\":\"ad")]          // admin or adult
    [InlineData("{\"name\":\"x\",\"tags\":[\"a\"")]
    [InlineData("{\"name\":\"x\",\"address\":{\"city\":\"y\"")]
    public void ValidPrefixesSurviveButAreIncomplete(string json)
    {
        var (accepted, state) = Feed(json);

        Assert.Equal(json.Length, accepted);
        Assert.False(state.IsComplete);
    }

    [Theory]
    // --- names -------------------------------------------------------------
    [InlineData("{\"zzz\"", 2)]                            // not a declared name
    [InlineData("{\"nx", 3)]                               // `n` narrows to two names, `x` to none
    [InlineData("{\"nam\"", 5)]                            // a prefix of `name`, not a name
    [InlineData("{\"names\"", 6)]                          // one character too many
    [InlineData("{\"name\":\"x\",\"name\"", 14)]           // already written
    [InlineData("{\"name\":\"x\",\"n\\u0061me\"", 14)]     // names are matched literally
    // --- types -------------------------------------------------------------
    [InlineData("{\"name\":1", 8)]                         // string declared
    [InlineData("{\"name\":true", 8)]
    [InlineData("{\"name\":{", 8)]
    [InlineData("{\"name\":[", 8)]
    [InlineData("{\"name\":null", 8)]
    [InlineData("{\"name\":\"x\",\"age\":\"7\"", 18)]      // integer declared
    [InlineData("{\"name\":\"x\",\"age\":1.", 19)]         // an integer has no fraction
    [InlineData("{\"name\":\"x\",\"age\":1e", 19)]         // and no exponent
    [InlineData("{\"name\":\"x\",\"active\":1", 21)]       // boolean declared
    [InlineData("{\"name\":\"x\",\"tags\":[1", 20)]        // an array of strings
    [InlineData("{\"name\":\"x\",\"tags\":{", 19)]         // array declared
    [InlineData("{\"name\":\"x\",\"address\":\"y\"", 22)]  // object declared
    // --- required ----------------------------------------------------------
    [InlineData("{}", 1)]                                  // name is required
    [InlineData("{\"age\":1}", 8)]
    [InlineData("{\"name\":\"x\",\"address\":{}", 23)]     // city is required
    [InlineData("{\"name\":\"x\",\"address\":{\"zip\":1}", 30)]
    // --- enums -------------------------------------------------------------
    [InlineData("{\"name\":\"x\",\"role\":\"adx", 22)]
    [InlineData("{\"name\":\"x\",\"role\":\"ad\"", 22)]    // a prefix of two members
    [InlineData("{\"name\":\"x\",\"role\":\"admins\"", 25)]
    [InlineData("{\"name\":\"x\",\"role\":\"USER", 20)]
    public void ViolationsDieAtTheOffendingCharacter(string json, int expectedAccepted)
    {
        var (accepted, _) = Feed(json);

        Assert.Equal(expectedAccepted, accepted);
    }

    /// A free-form value is checked as JSON and nothing more, and control returns
    /// to the schema when it closes.
    [Fact]
    public void AFreeFormValueEscapesTheSchemaAndHandsBackControl()
    {
        var (accepted, state) = Feed("{\"meta\":{\"zzz\":1},\"nope\"");

        Assert.Equal(20, accepted);        // `zzz` was fine inside meta; `no` matches no name
        Assert.False(state.IsComplete);
    }

    [Fact]
    public void AnObjectWhoseEveryNameIsWrittenCannotOpenAnother()
    {
        // Every declared name of `address` is used, so a third key is impossible.
        var (accepted, _) = Feed("{\"name\":\"x\",\"address\":{\"city\":\"y\",\"zip\":1,\"");

        Assert.Equal(42, accepted);
    }

    [Fact]
    public void AMaskCanBeGivenASchema()
    {
        string[] vocab = ["{", "}", "\"name\":", "\"other\":", "\"x\"", "<eos>"];
        var mask = new JsonMask(vocab, endOfSequence: 5, JsonSchema.Parse(Schema));
        var logits = new float[vocab.Length];

        mask.Commit(0);                   // `{`
        mask.Apply(logits);

        Assert.False(float.IsNegativeInfinity(logits[2]));   // the only declared name
        Assert.True(float.IsNegativeInfinity(logits[3]));
        Assert.True(float.IsNegativeInfinity(logits[1]));    // name is required
    }

    [Fact]
    public void ResetKeepsTheSchema()
    {
        string[] vocab = ["{", "}", "\"name\":", "\"other\":", "\"x\"", "<eos>"];
        var mask = new JsonMask(vocab, endOfSequence: 5, JsonSchema.Parse(Schema));
        var logits = new float[vocab.Length];

        mask.Commit(0);
        mask.Reset();
        mask.Commit(0);
        mask.Apply(logits);

        Assert.True(float.IsNegativeInfinity(logits[3]));
    }

    [Fact]
    public void WithoutASchemaAnyShapeIsStillAccepted()
    {
        var state = new JsonState();

        foreach (var c in "{\"zzz\":[1]}")
            Assert.True(state.TryAdvance(c));

        Assert.True(state.IsComplete);
    }

    // --- compiling ---------------------------------------------------------

    [Theory]
    [InlineData("[]")]                                                  // not an object
    [InlineData("{\"type\":\"string\"}")]                               // root is not an object
    [InlineData("{\"type\":\"object\"}")]                               // no properties
    [InlineData("{\"properties\":{\"a\":{\"type\":\"bogus\"}}}")]
    [InlineData("{\"properties\":{\"a\":{}},\"required\":[\"b\"]}")]    // b is not declared
    public void InvalidSchemasAreRefused(string schema)
    {
        Assert.Throws<ArgumentException>(() => JsonSchema.Parse(schema));
    }

    [Fact]
    public void ASchemaNestedTooDeeplyIsRefused()
    {
        // One object schema per level, one level past the limit.
        var schema = "{\"type\":\"string\"}";
        for (var i = 0; i < JsonSchema.MaxDepth + 1; i++)
            schema = "{\"type\":\"object\",\"properties\":{\"a\":" + schema + "}}";

        Assert.Throws<ArgumentException>(() => JsonSchema.Parse(schema));
    }

    [Fact]
    public void MoreThanSixtyFourPropertiesIsRefused()
    {
        var properties = Enumerable.Range(0, JsonSchema.MaxProperties + 1)
            .Select(i => $"\"p{i}\":{{\"type\":\"string\"}}");

        var schema = "{\"properties\":{" + string.Join(",", properties) + "}}";

        Assert.Throws<ArgumentException>(() => JsonSchema.Parse(schema));
    }

    [Fact]
    public void AnUndeclaredTypeAcceptsAnyValue()
    {
        var schema = JsonSchema.Parse("{\"properties\":{\"a\":{}}}");

        foreach (var json in new[] { "{\"a\":1}", "{\"a\":\"s\"}", "{\"a\":[true]}", "{\"a\":{\"b\":0}}" })
        {
            var state = new JsonState(schema);
            foreach (var c in json)
                Assert.True(state.TryAdvance(c), $"rejected {json}");

            Assert.True(state.IsComplete, $"never finished {json}");
        }
    }

    /// The property that matters: a document the schema allows must never have a
    /// prefix rejected, and must finish exactly at its last character.
    [Fact]
    public void AcceptsEveryPrefixOfEveryMatchingDocument()
    {
        var random = new Random(20260924);

        for (var i = 0; i < 500; i++)
        {
            var json = RandomMatchingDocument(random);
            Assert.True(Parses(json), $"generator produced invalid JSON: {json}");

            var state = new JsonState(Compiled);
            for (var c = 0; c < json.Length; c++)
            {
                Assert.True(state.TryAdvance(json[c]),
                    $"rejected a valid prefix at index {c} of: {json}");

                Assert.False(state.IsComplete && c < json.Length - 1,
                    $"finished early at index {c} of: {json}");
            }

            Assert.True(state.IsComplete, $"never finished: {json}");
        }
    }

    /// A random document matching Schema: `name` always, then a random subset of
    /// the rest in a random order.
    static string RandomMatchingDocument(Random random)
    {
        string[] names = ["nickname", "age", "score", "active", "role", "tags", "address", "meta"];

        var fields = new List<string> { "\"name\":\"n\"" };
        foreach (var name in names.OrderBy(_ => random.Next()))
            if (random.Next(2) == 0)
                fields.Add($"\"{name}\":{RandomValueFor(name, random)}");

        return "{" + string.Join(",", fields) + "}";
    }

    static string RandomValueFor(string name, Random random) => name switch
    {
        "nickname" => "\"nick\"",
        "age" => (random.Next(2) == 0 ? "-" : "") + random.Next(0, 500),
        "score" => random.Next(3) switch
        {
            0 => "0",
            1 => "12.5",
            _ => "-3e+2",
        },
        "active" => random.Next(2) == 0 ? "true" : "false",
        "role" => "\"" + new[] { "admin", "adult", "user" }[random.Next(3)] + "\"",
        "tags" => "[" + string.Join(",", Enumerable.Range(0, random.Next(3)).Select(t => $"\"t{t}\"")) + "]",
        "address" => random.Next(2) == 0
            ? "{\"city\":\"c\"}"
            : "{\"zip\":9,\"city\":\"c\"}",
        _ => random.Next(2) == 0 ? "{}" : "{\"free\":[1,null,{\"x\":\"y\"}]}",
    };

    static bool Parses(string json)
    {
        try { JsonDocument.Parse(json); return true; }
        catch (JsonException) { return false; }
    }
}
