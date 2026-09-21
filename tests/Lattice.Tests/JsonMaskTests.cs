using Lattice;
using Xunit;

namespace Lattice.Tests;

public class JsonMaskTests
{
    // A hand-built vocabulary. The interesting entries are the ones that span
    // JSON boundaries — `":` and `,"` each cross from one expectation into the
    // next, which is the case a position-by-position checker would get wrong.
    const int Brace = 0, CloseBrace = 1, Quote = 2, QuoteA = 3, QuoteColon = 4,
              CommaQuote = 5, CommaBracket = 6, One = 7, Hello = 8, Fence = 9,
              Empty = 10, Eos = 11;

    static readonly string[] Vocab =
    [
        "{",      // 0
        "}",      // 1
        "\"",     // 2
        "\"a",    // 3
        "\":",    // 4
        ",\"",    // 5
        ",]",     // 6
        "1",      // 7
        "hello",  // 8
        "```",    // 9
        "",       // 10
        "<eos>",  // 11
    ];

    static JsonMask NewMask() => new(Vocab, Eos);

    /// The ids left standing after the mask runs.
    static int[] Legal(JsonMask mask)
    {
        var logits = new float[Vocab.Length];
        mask.Apply(logits);

        return Enumerable.Range(0, logits.Length)
            .Where(id => !float.IsNegativeInfinity(logits[id]))
            .ToArray();
    }

    static JsonMask MaskAfter(params int[] tokens)
    {
        var mask = NewMask();
        foreach (var token in tokens) mask.Commit(token);
        return mask;
    }

    [Fact]
    public void AtTheStartOnlyAnOpeningBraceSurvives()
    {
        Assert.Equal([Brace], Legal(NewMask()));
    }

    [Fact]
    public void ProseAndMarkdownFencesAreNeverLegal()
    {
        var mask = MaskAfter(Brace);   // expecting a key or `}`

        var legal = Legal(mask);

        Assert.DoesNotContain(Fence, legal);
        Assert.DoesNotContain(Hello, legal);
        Assert.Contains(Quote, legal);
        Assert.Contains(CloseBrace, legal);
    }

    [Fact]
    public void AnEmptyTokenIsNeverLegal()
    {
        Assert.DoesNotContain(Empty, Legal(NewMask()));
        Assert.DoesNotContain(Empty, Legal(MaskAfter(Brace)));
    }

    /// The tokenizer-boundary case. After `{"a` the token `":` carries the
    /// closing quote of the key *and* the colon, crossing two expectations in
    /// one token.
    [Fact]
    public void ATokenMaySpanTwoExpectations()
    {
        var mask = MaskAfter(Brace, QuoteA);

        Assert.Contains(QuoteColon, Legal(mask));
    }

    /// Same idea, now where the second half decides it: from inside a number
    /// in an object, `,"` is legal (the number ends, a key begins) while `,]`
    /// is not (there is no array to close). The first character is identical.
    [Fact]
    public void ATokensSecondCharacterCanMakeItIllegal()
    {
        var mask = MaskAfter(Brace, QuoteA, QuoteColon, One);

        var legal = Legal(mask);

        Assert.Contains(CommaQuote, legal);
        Assert.DoesNotContain(CommaBracket, legal);
    }

    [Fact]
    public void EndOfSequenceIsIllegalUntilTheObjectCloses()
    {
        Assert.DoesNotContain(Eos, Legal(NewMask()));
        Assert.DoesNotContain(Eos, Legal(MaskAfter(Brace)));
        Assert.DoesNotContain(Eos, Legal(MaskAfter(Brace, QuoteA, QuoteColon, One)));
    }

    [Fact]
    public void OnceCompleteOnlyEndOfSequenceSurvives()
    {
        var mask = MaskAfter(Brace, CloseBrace);

        Assert.True(mask.IsComplete);
        Assert.Equal([Eos], Legal(mask));
    }

    [Fact]
    public void CommittingEndOfSequenceDoesNotDisturbTheState()
    {
        var mask = MaskAfter(Brace, CloseBrace);

        mask.Commit(Eos);

        Assert.True(mask.IsComplete);
    }

    [Fact]
    public void CommittingATokenTheMaskRuledOutThrows()
    {
        var mask = NewMask();

        Assert.Throws<InvalidOperationException>(() => mask.Commit(Hello));
    }

    [Fact]
    public void ResetStartsAFreshDocument()
    {
        var mask = MaskAfter(Brace, CloseBrace);

        mask.Reset();

        Assert.False(mask.IsComplete);
        Assert.Equal([Brace], Legal(mask));
    }

    [Fact]
    public void ApplyLeavesLogitsAlreadyRuledOutUpstream()
    {
        var mask = NewMask();
        var logits = new float[Vocab.Length];
        logits[Brace] = float.NegativeInfinity;   // the only otherwise-legal token

        mask.Apply(logits);

        Assert.All(logits, logit => Assert.True(float.IsNegativeInfinity(logit)));
    }

    [Fact]
    public void ApplyRejectsALogitBufferSmallerThanTheVocabulary()
    {
        var mask = NewMask();

        Assert.Throws<ArgumentException>(() => mask.Apply(new float[Vocab.Length - 1]));
    }
}
