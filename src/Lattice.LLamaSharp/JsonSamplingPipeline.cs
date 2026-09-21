using LLama;
using LLama.Native;
using LLama.Sampling;

namespace Lattice.LLamaSharp;

/// A sampling pipeline that constrains generation to a single JSON object.
///
/// Before each token is chosen, every token that would break the JSON has its
/// logit set to negative infinity; the remaining tokens are then sampled
/// normally, so the model keeps its own preferences among the legal ones.
public sealed class JsonSamplingPipeline : ISamplingPipeline
{
    readonly DefaultSamplingPipeline _inner = new();
    readonly JsonMask _mask;

    /// <param name="weights">The model whose vocabulary the mask is built from.</param>
    public JsonSamplingPipeline(LLamaWeights weights)
    {
        ArgumentNullException.ThrowIfNull(weights);

        var eos = weights.Vocab.EOS ?? throw new ArgumentException(
            "the model has no end-of-sequence token", nameof(weights));

        _mask = new JsonMask(DecodeVocabulary(weights), (int)eos);
    }

    /// What the next character may be.
    public Expect Expecting => _mask.Expecting;

    /// True once the object has closed.
    public bool IsComplete => _mask.IsComplete;

    public LLamaToken Sample(SafeLLamaContextHandle ctx, int index)
    {
        _mask.Apply(ctx.GetLogitsIth(index));

        var token = _inner.Sample(ctx, index);

        // Advance here rather than in Accept: StatelessExecutor never calls Accept.
        _mask.Commit((int)token);

        return token;
    }

    public void Accept(LLamaToken token) => _inner.Accept(token);

    public void Apply(SafeLLamaContextHandle ctx, LLamaTokenDataArray data) => _inner.Apply(ctx, data);

    /// Starts a fresh document.
    public void Reset()
    {
        _mask.Reset();
        _inner.Reset();
    }

    public void Dispose() => _inner.Dispose();

    /// Every token id's text, indexed by id.
    static string[] DecodeVocabulary(LLamaWeights weights)
    {
        var vocab = weights.Vocab;
        var text = new string[vocab.Count];

        for (var id = 0; id < text.Length; id++)
            text[id] = vocab.LLamaTokenToString((LLamaToken)id, true) ?? "";

        return text;
    }
}
