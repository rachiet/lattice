namespace Lattice;

/// Decides which vocabulary tokens are legal at the current point in a JSON
/// document, and writes that decision into a logit buffer.
///
/// Tokens are tested character by character, so a token whose text spans a JSON
/// boundary — `":` or `,"` — is handled like any other.
public sealed class JsonMask
{
    readonly string[] _vocab;
    readonly int _endOfSequence;
    JsonState _state;

    /// <param name="vocab">Every token id's text, indexed by id. Held by
    /// reference and never modified.</param>
    /// <param name="endOfSequence">The token that ends generation.</param>
    public JsonMask(string[] vocab, int endOfSequence)
    {
        ArgumentNullException.ThrowIfNull(vocab);
        if (endOfSequence < 0 || endOfSequence >= vocab.Length)
            throw new ArgumentOutOfRangeException(nameof(endOfSequence));

        _vocab = vocab;
        _endOfSequence = endOfSequence;
    }

    /// True once the object has closed, after which the end-of-sequence token
    /// is the only legal one.
    public bool IsComplete => _state.IsComplete;

    /// What the next character may be.
    public Expect Expecting => _state.Expecting;

    /// Sets the logit of every token that would break the JSON to negative
    /// infinity, and leaves the rest untouched.
    public void Apply(Span<float> logits)
    {
        if (logits.Length < _vocab.Length)
            throw new ArgumentException(
                $"logits has {logits.Length} entries, vocabulary has {_vocab.Length}", nameof(logits));

        // Finished: stopping is the only legal move.
        if (_state.IsComplete)
        {
            for (var id = 0; id < logits.Length; id++)
                if (id != _endOfSequence)
                    logits[id] = float.NegativeInfinity;
            return;
        }

        for (var id = 0; id < logits.Length; id++)
        {
            if (float.IsNegativeInfinity(logits[id])) continue;  // already ruled out

            // Stopping before the object closes would truncate the document.
            if (id == _endOfSequence || !IsLegal(_vocab[id]))
                logits[id] = float.NegativeInfinity;
        }
    }

    /// Advances the state by the token that was sampled. Throws if that token
    /// is not legal at this point.
    public void Commit(int tokenId)
    {
        if (tokenId == _endOfSequence) return;

        foreach (var c in _vocab[tokenId])
            if (!_state.TryAdvance(c))
                throw new InvalidOperationException(
                    $"token {tokenId} ({_vocab[tokenId]}) is not legal here");
    }

    /// Starts a fresh document.
    public void Reset() => _state = default;

    /// A token is legal when every one of its characters is. The probe is a
    /// copy, so a rejection leaves the committed state untouched.
    bool IsLegal(string text)
    {
        // An empty token would pass without advancing the state.
        if (text.Length == 0) return false;

        var probe = _state;
        foreach (var c in text)
            if (!probe.TryAdvance(c))
                return false;

        return true;
    }
}
