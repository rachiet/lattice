namespace Lattice;

/// What the next character is allowed to be, given everything read so far.
///
/// Inside a number, a `*Digit` name owes a character and cannot end the number;
/// a `More*` name may either continue or end it.
public enum Expect
{
    /// Only `{`. The document must be a JSON object, so `123`, `[1]` and `"hi"`
    /// are refused at the first character. The zero value, so a new JsonState
    /// starts here.
    Object,

    /// A value must start here: after `:`, or after `,` in an array.
    Value,
    /// A value, or `]` to close an empty array. Reached from `[`.
    ValueOrClose,
    /// A key's opening quote, or `}` to close an empty object. Reached from `{`.
    KeyOrClose,
    /// A key's opening quote. No `}` — `{"a":1,}` is invalid. Reached from a
    /// `,` inside an object.
    Key,
    /// More of the key, or the `"` that closes it.
    KeyText,
    /// Only `:`. Reached from a key's closing quote.
    Colon,
    /// More of the string, or the `"` that closes it.
    StringText,
    /// A `,`, or the closer matching the innermost container. Reached whenever
    /// a value finishes.
    CommaOrClose,
    /// Only a digit. Reached from `-`.
    FirstDigit,
    /// `.`, `e`/`E`, or the number ends here. No further digit — that would be
    /// a leading zero. Reached from a `0`.
    FractionOrExponent,
    /// More digits, `.`, `e`/`E`, or the number ends here.
    MoreInteger,
    /// Only a digit. Reached from `.`.
    FractionDigit,
    /// More digits, `e`/`E`, or the number ends here.
    MoreFraction,
    /// `+`, `-`, or a digit. Reached from `e`/`E`.
    ExponentSignOrDigit,
    /// Only a digit. Reached from `+` or `-`.
    ExponentDigit,
    /// More digits, or the number ends here.
    MoreExponent,
    /// The next letter of `true`, `false` or `null`, and nothing else.
    LiteralLetter,
    /// Only whitespace. The object closed and depth is back to 0.
    End,
}

/// A character-level JSON checker: does this character keep the document a
/// valid JSON prefix, and is that prefix now complete?
///
/// All fields are value types, so copying a JsonState copies the whole state.
/// Open containers are packed into a ulong bitmask, one bit each.
public struct JsonState
{
    /// Objects and arrays nested deeper than this are rejected.
    public const int MaxDepth = 64;

    Expect _expect;
    ulong _containers;  // one bit per open container, innermost in bit (_depth - 1): 1 = array, 0 = object
    int _depth;
    bool _escaped;      // the previous character was a backslash inside a string
    int _hexLeft;       // remaining hex digits owed to a \u escape
    byte _literal;      // which word is being matched: 1 = true, 2 = false, 3 = null
    byte _literalIndex; // how much of that word has been consumed

    public readonly Expect Expecting => _expect;
    public readonly int Depth => _depth;

    /// True once the object has closed — the `}` that takes depth back to 0.
    public readonly bool IsComplete => _expect == Expect.End;

    /// Feeds one character. Returns true if the document is still valid and the
    /// state has advanced; false if the character breaks it, in which case the
    /// state may have partly advanced and must be discarded.
    public bool TryAdvance(char c) => _expect switch
    {
        Expect.KeyText or Expect.StringText => AdvanceString(c),

        Expect.Object => IsSpace(c) || c == '{' && Push(array: false, Expect.KeyOrClose),

        Expect.Value => IsSpace(c) || BeginValue(c),

        Expect.ValueOrClose => IsSpace(c) || c switch
        {
            ']' => Pop(array: true),
            _ => BeginValue(c),
        },

        Expect.KeyOrClose => IsSpace(c) || c switch
        {
            '"' => Enter(Expect.KeyText),
            '}' => Pop(array: false),
            _ => false,
        },

        Expect.Key => IsSpace(c) || c == '"' && Enter(Expect.KeyText),

        Expect.Colon => IsSpace(c) || c == ':' && Enter(Expect.Value),

        Expect.CommaOrClose => AdvanceCommaOrClose(c),

        // A number has no closing character: it ends when a delimiter arrives,
        // which is then re-read from CommaOrClose.
        Expect.FirstDigit => c == '0' ? Enter(Expect.FractionOrExponent)
                           : IsDigit19(c) && Enter(Expect.MoreInteger),

        Expect.FractionOrExponent => c switch
        {
            '.' => Enter(Expect.FractionDigit),
            'e' or 'E' => Enter(Expect.ExponentSignOrDigit),
            _ => EndNumber(c),  // a second digit would be a leading zero
        },

        Expect.MoreInteger => IsDigit(c) || c switch
        {
            '.' => Enter(Expect.FractionDigit),
            'e' or 'E' => Enter(Expect.ExponentSignOrDigit),
            _ => EndNumber(c),
        },

        Expect.FractionDigit => IsDigit(c) && Enter(Expect.MoreFraction),

        Expect.MoreFraction => IsDigit(c) || c switch
        {
            'e' or 'E' => Enter(Expect.ExponentSignOrDigit),
            _ => EndNumber(c),
        },

        Expect.ExponentSignOrDigit => c is '+' or '-' ? Enter(Expect.ExponentDigit)
                         : IsDigit(c) && Enter(Expect.MoreExponent),

        Expect.ExponentDigit => IsDigit(c) && Enter(Expect.MoreExponent),

        Expect.MoreExponent => IsDigit(c) || EndNumber(c),

        Expect.LiteralLetter => AdvanceLiteral(c),

        Expect.End => IsSpace(c),

        _ => false,
    };

    bool AdvanceCommaOrClose(char c)
    {
        if (IsSpace(c)) return true;
        return c switch
        {
            ',' when _depth > 0 => Enter(InnermostIsArray() ? Expect.Value : Expect.Key),
            '}' => Pop(array: false),
            ']' => Pop(array: true),
            _ => false,
        };
    }

    bool AdvanceString(char c)
    {
        // \u must be followed by exactly four hex digits before anything else.
        if (_hexLeft > 0)
        {
            if (!IsHex(c)) return false;
            _hexLeft--;
            return true;
        }

        if (_escaped)
        {
            if (c == 'u') _hexLeft = 4;
            else if (c is not ('"' or '\\' or '/' or 'b' or 'f' or 'n' or 'r' or 't')) return false;
            _escaped = false;
            return true;
        }

        if (c == '\\') { _escaped = true; return true; }

        // A quote closes the key or the string value.
        if (c == '"') return _expect == Expect.KeyText ? Enter(Expect.Colon) : EndValue();

        return c >= 0x20;  // raw control characters must be escaped
    }

    bool AdvanceLiteral(char c)
    {
        var word = _literal switch { 1 => "true", 2 => "false", _ => "null" };
        if (c != word[_literalIndex]) return false;
        _literalIndex++;
        return _literalIndex == word.Length ? EndValue() : true;
    }

    bool BeginValue(char c)
    {
        switch (c)
        {
            case '{': return Push(array: false, Expect.KeyOrClose);
            case '[': return Push(array: true, Expect.ValueOrClose);
            case '"': return Enter(Expect.StringText);
            case '-': return Enter(Expect.FirstDigit);
            case '0': return Enter(Expect.FractionOrExponent);
            case 't': return BeginLiteral(1);
            case 'f': return BeginLiteral(2);
            case 'n': return BeginLiteral(3);
            default: return IsDigit19(c) && Enter(Expect.MoreInteger);
        }
    }

    bool BeginLiteral(byte literal)
    {
        _literal = literal;
        _literalIndex = 1;  // the first letter is the character we just consumed
        _expect = Expect.LiteralLetter;
        return true;
    }

    bool Push(bool array, Expect next)
    {
        if (_depth == MaxDepth) return false;
        if (array) _containers |= 1UL << _depth;
        else _containers &= ~(1UL << _depth);
        _depth++;
        _expect = next;
        return true;
    }

    bool Pop(bool array)
    {
        if (_depth == 0 || InnermostIsArray() != array) return false;
        _depth--;
        return EndValue();
    }

    /// A value just completed. At depth 0 that was the whole document.
    bool EndValue()
    {
        _expect = _depth == 0 ? Expect.End : Expect.CommaOrClose;
        return true;
    }

    /// Close the number, then let CommaOrClose (or End) rule on the delimiter.
    /// Terminates: EndValue never lands back on a number position.
    bool EndNumber(char c)
    {
        EndValue();
        return TryAdvance(c);
    }

    bool Enter(Expect next)
    {
        _expect = next;
        return true;
    }

    readonly bool InnermostIsArray() => (_containers & 1UL << _depth - 1) != 0;

    static bool IsSpace(char c) => c is ' ' or '\t' or '\n' or '\r';
    static bool IsDigit(char c) => c is >= '0' and <= '9';
    static bool IsDigit19(char c) => c is >= '1' and <= '9';
    static bool IsHex(char c) => IsDigit(c) || c is >= 'a' and <= 'f' or >= 'A' and <= 'F';
}
