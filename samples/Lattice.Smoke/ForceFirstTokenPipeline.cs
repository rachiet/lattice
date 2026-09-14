using LLama.Native;
using LLama.Sampling;

// Proof that we control token selection: on the first step every token except
// `forced` gets a logit of -infinity. After that, sampling is left untouched.
sealed class ForceFirstTokenPipeline(LLamaToken forced) : ISamplingPipeline
{
    readonly DefaultSamplingPipeline _inner = new();
    bool _first = true;

    public LLamaToken Sample(SafeLLamaContextHandle ctx, int index)
    {
        if (_first)
        {
            // A view over llama.cpp's own logits buffer; the inner sampler reads the same memory.
            var logits = ctx.GetLogitsIth(index);
            for (var id = 0; id < logits.Length; id++)
                if (id != (int)forced)
                    logits[id] = float.NegativeInfinity;
            // Flip here, not in Accept: StatelessExecutor never calls Accept.
            _first = false;
        }
        return _inner.Sample(ctx, index);
    }

    public void Accept(LLamaToken token) => _inner.Accept(token);

    public void Apply(SafeLLamaContextHandle ctx, LLamaTokenDataArray data) => _inner.Apply(ctx, data);

    public void Reset()
    {
        _first = true;
        _inner.Reset();
    }

    public void Dispose() => _inner.Dispose();
}
