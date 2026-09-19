using System.Text;
using System.Text.Json;
using Lattice;
using Xunit;

namespace Lattice.Tests;

public class JsonStateTests
{
    /// Feeds a string character by character. Returns how many characters were
    /// accepted before one was rejected, and the state it ended in.
    ///
    /// A plain foreach, deliberately: JsonState is a struct, so passing
    /// state.TryAdvance as a delegate would bind to a boxed copy and every
    /// transition would be thrown away.
    static (int Accepted, JsonState State) Feed(string json)
    {
        var state = new JsonState();
        var accepted = 0;
        foreach (var c in json)
        {
            if (!state.TryAdvance(c)) break;
            accepted++;
        }
        return (accepted, state);
    }

    [Theory]
    // The contract: the document is a JSON object. Everything else is refused
    // at character 0, even though .NET would parse it happily.
    [InlineData("{}", 2, true)]
    [InlineData("  {}", 4, true)]
    [InlineData("{\"a\":1}", 7, true)]
    [InlineData("{\"a\":1", 6, false)]   // a value, but not a document
    [InlineData("123", 0, false)]
    [InlineData("[1]", 0, false)]
    [InlineData("\"hi\"", 0, false)]
    [InlineData("true", 0, false)]
    [InlineData("null", 0, false)]
    public void RootMustBeAnObject(string json, int expectedAccepted, bool complete)
    {
        var (accepted, state) = Feed(json);

        Assert.Equal(expectedAccepted, accepted);
        Assert.Equal(complete, state.IsComplete);
    }

    [Theory]
    // --- valid prefixes: still alive, not yet a document -------------------
    // Values live inside the object, which is the only place they can appear.
    [InlineData("{")]
    [InlineData("{\"")]
    [InlineData("{\"a")]
    [InlineData("{\"a\"")]
    [InlineData("{\"a\":")]
    [InlineData("{\"a\":1")]              // a number is unfinished until a delimiter arrives
    [InlineData("{\"a\":1,")]
    [InlineData("{\"a\":{\"b\":[")]
    [InlineData("{\"a\":[")]
    [InlineData("{\"a\":[1,")]
    [InlineData("{\"a\":[1,2")]
    [InlineData("{\"a\":-")]
    [InlineData("{\"a\":1.")]
    [InlineData("{\"a\":1e")]
    [InlineData("{\"a\":1e+")]
    [InlineData("{\"a\":\"unterminated")]
    [InlineData("{\"a\":\"esc\\")]
    [InlineData("{\"a\":\"esc\\u")]
    [InlineData("{\"a\":\"esc\\u00")]
    [InlineData("{\"a\":tru")]
    [InlineData("  {  \"a\"  :  ")]
    public void ValidPrefixesSurviveButAreIncomplete(string json)
    {
        var (accepted, state) = Feed(json);

        Assert.Equal(json.Length, accepted);
        Assert.False(state.IsComplete);
    }

    [Theory]
    // --- complete documents: alive and finished ---------------------------
    [InlineData("{}")]
    [InlineData("{\"a\":1}")]
    [InlineData("{\"a\":[1,{\"b\":null}]}")]
    [InlineData("{\"a\":[[[]]]}")]
    [InlineData("{\"a\":\"str\"}")]
    [InlineData("{\"a\":\"\\u00e9\\n\\\\\"}")]
    [InlineData("{\"a\":123}")]
    [InlineData("{\"a\":-0.5e-10}")]
    [InlineData("{\"a\":0}")]
    [InlineData("{\"a\":true,\"b\":false,\"c\":null}")]
    [InlineData("{\"a\":1} ")]            // trailing whitespace is still legal
    public void CompleteDocumentsAreMarkedComplete(string json)
    {
        var (accepted, state) = Feed(json);

        Assert.Equal(json.Length, accepted);
        Assert.True(state.IsComplete);
        Assert.True(Parses(json));        // and .NET agrees it is a document
    }

    [Theory]
    // --- invalid: must die, and at exactly this character index -----------
    [InlineData("{,", 1)]                       // a key must come first
    [InlineData("{\"a\":,", 5)]                 // a value must come first
    [InlineData("{\"a\":1,}", 7)]               // trailing comma in an object
    [InlineData("{\"a\":[1,]}", 8)]             // trailing comma in an array
    [InlineData("{\"a\"1}", 4)]                 // colon missing
    [InlineData("{\"a\":1]}", 6)]               // closer does not match the container
    [InlineData("{\"a\":[1}]", 7)]
    [InlineData("}", 0)]
    [InlineData("{'a':1}", 1)]                  // single quotes are not JSON
    [InlineData("{\"a\":01}", 6)]               // leading zero
    [InlineData("{\"a\":1..2}", 7)]
    [InlineData("{\"a\":1e++1}", 8)]
    [InlineData("{\"a\":tru3}", 8)]
    [InlineData("{\"a\":nulll}", 9)]            // the value ended at "null"
    [InlineData("{\"a\":\"bad\\q\"}", 10)]      // not an escape
    [InlineData("{\"a\":\"bad\\u00zz\"}", 13)]  // not hex
    [InlineData("{\"a\":\"raw\nnewline\"}", 9)] // control characters must be escaped
    [InlineData("{}{}", 2)]                     // one document only
    [InlineData("```json", 0)]                  // the failure that broke plain sampling
    [InlineData("Here is a cat:", 0)]
    public void InvalidInputDiesAtTheOffendingCharacter(string json, int expectedAccepted)
    {
        var (accepted, _) = Feed(json);

        Assert.Equal(expectedAccepted, accepted);
        Assert.False(Parses(json));
    }

    [Fact]
    public void NestingDeeperThanMaxDepthIsRejected()
    {
        const string prefix = "{\"a\":";   // the object itself is depth 1

        var (accepted, _) = Feed(prefix + new string('[', JsonState.MaxDepth));

        Assert.Equal(prefix.Length + JsonState.MaxDepth - 1, accepted);
    }

    /// The property that matters, and the one the hand-written tables will
    /// always be missing a case from: if .NET can parse it, we must never
    /// reject any prefix of it, and we must call it complete exactly at the end.
    [Fact]
    public void AcceptsEveryPrefixOfValidJson()
    {
        var random = new Random(20260915);

        for (var i = 0; i < 500; i++)
        {
            var json = RandomObject(random, depth: 0);
            Assert.True(Parses(json), $"generator produced invalid JSON: {json}");

            var state = new JsonState();
            for (var c = 0; c < json.Length; c++)
            {
                Assert.True(state.TryAdvance(json[c]),
                    $"rejected a valid prefix at index {c} of: {json}");

                // Only the final `}` may finish the document.
                Assert.False(state.IsComplete && c < json.Length - 1,
                    $"finished early at index {c} of: {json}");
            }

            Assert.True(state.IsComplete, $"never finished: {json}");
        }
    }

    static string RandomValue(Random random, int depth) => random.Next(depth < 3 ? 8 : 6) switch
    {
        0 => "null",
        1 => random.Next(2) == 0 ? "true" : "false",
        2 or 3 => RandomNumber(random),
        4 or 5 => RandomString(random),
        6 => RandomArray(random, depth),
        _ => RandomObject(random, depth),
    };

    static string RandomArray(Random random, int depth)
    {
        var items = Enumerable.Range(0, random.Next(4)).Select(_ => RandomValue(random, depth + 1));
        return $"[{string.Join(",", items)}]";
    }

    static string RandomObject(Random random, int depth)
    {
        var fields = Enumerable.Range(0, random.Next(4))
            .Select(_ => $"{RandomString(random)}:{RandomValue(random, depth + 1)}");
        return $"{{{string.Join(",", fields)}}}";
    }

    static string RandomNumber(Random random)
    {
        var text = new StringBuilder();
        if (random.Next(2) == 0) text.Append('-');
        text.Append(random.Next(2) == 0 ? "0" : random.Next(1, 10).ToString());
        if (random.Next(2) == 0) text.Append('.').Append(random.Next(0, 1000));
        if (random.Next(3) == 0) text.Append(random.Next(2) == 0 ? 'e' : 'E')
                                     .Append("+-"[random.Next(2)])
                                     .Append(random.Next(0, 30));
        return text.ToString();
    }

    /// Leans on the characters that have their own branches in AdvanceString:
    /// the escape table, \u, and plain letters.
    static string RandomString(Random random)
    {
        string[] pieces = ["a", "Z", " ", "\\\"", "\\\\", "\\/", "\\n", "\\t", "\\b", "\\u00e9", "\\uD83D"];
        var text = new StringBuilder("\"");
        for (var i = random.Next(6); i > 0; i--)
            text.Append(pieces[random.Next(pieces.Length)]);
        return text.Append('"').ToString();
    }

    static bool Parses(string json)
    {
        try { JsonDocument.Parse(json); return true; }
        catch (JsonException) { return false; }
    }
}
