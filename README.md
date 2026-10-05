# Lattice

A constrained decoding engine for .NET. It makes schema-invalid LLM output
impossible rather than unlikely.

At each generation step a local model emits a logit for every token in its
vocabulary (~151k for Qwen2.5). Before a token is sampled, Lattice sets the
logit of every token that would break the JSON — or violate your JSON Schema —
to negative infinity. The model keeps its freedom over *what* it says and loses
all freedom over *shape*. There is no parse-and-retry loop, because the first
attempt cannot be malformed.

All the work happens between "logits produced" and "token chosen", which means
it needs local weights: a hosted API returns finished text, and there are no
logits left to mask.

## How it works

```
once:
    schema text ──► JsonSchema.Parse ──► one node per declared object:
                                         names as a prefix table, types,
                                         required mask, enum tables
    model       ──► vocabulary ────────► JsonMask(vocab, eos, schema)

per generated token:

    logits[151936]  ◄── the model
         │
         ▼
    JsonMask.Apply
         │
         ├─ for every token id: copy JsonState, push the token's
         │  characters through TryAdvance
         │
         └─ any character refused ──► logits[id] = -infinity
         │
         ▼
    the sampler picks one of the survivors   ← temperature, top-k, top-p
         │                                     all still apply
         ▼
    JsonMask.Commit ──► the same characters advance the real JsonState
         │
         └──► repeat until IsComplete, after which only the
              end-of-sequence token survives
```

`TryAdvance` is the whole engine, and it answers two questions per character.
Is this still valid JSON — brace matching, number grammar, string escapes. And,
when a schema is present, does it still match the schema *at this position*.

Tokens do not align to JSON boundaries: a single token may be `":` or `,"` or
`"}\n`. Because the check is per character, a token spanning two grammar
positions is handled like any other.

### Matching property names

The part worth knowing is how names are enforced. Each declared object compiles
to a table answering "which of my property names have character *c* at position
*p*", as a bitmask with one bit per name. Checking a name is then a bitwise
`and` per character against the names not yet written.

For example, say the schema declares `name`, `nickname` and `age`. The model
has written `{"name": "Tom", ` and is now starting the next key. Each row is one
more character of that key:

```
key so far     names still possible    legal next character
"              nickname, age           n or a     ← "name" is used up
"n             nickname                i
"ni            nickname                c
...
"nickname      nickname (full match)   "          ← only the closing quote
"nickz         none                    —          ← z is masked: no name fits
```

Two things fall out for free. A name already written is absent from the
candidate set, so duplicate keys are impossible. And the closing quote is legal
only where exactly one declared name ends, so the key resolves to a single
property — which fixes the type of the value before a single character of it is
generated.

## Using it

With LLamaSharp:

```csharp
using LLama;
using LLama.Common;
using Lattice;
using Lattice.LLamaSharp;

var parameters = new ModelParams("models/qwen2.5-0.5b-instruct-q4_k_m.gguf")  // your GGUF model
{
    ContextSize = 2048,   // prompt + output; up to what your model was trained for
    GpuLayerCount = 99,   // layers on the GPU; 0 for CPU only
};

using var weights = LLamaWeights.LoadFromFile(parameters);

var schema = JsonSchema.Parse(File.ReadAllText("schemas/cat.json"));  // your JSON Schema
using var pipeline = new JsonSamplingPipeline(weights, schema);

var inferenceParams = new InferenceParams
{
    MaxTokens = 300,               // the most tokens to generate
    SamplingPipeline = pipeline,   // this overrides all other sampling
};

var prompt = "...";   // your prompt, in the model's chat format, with the schema in it

await foreach (var piece in new StatelessExecutor(weights, parameters).InferAsync(prompt, inferenceParams))
    Console.Write(piece);
```

The schema is optional. Create the pipeline without one,
`new JsonSamplingPipeline(weights)`, and Lattice still guarantees the output is
a single well-formed JSON object, but with any keys and values.

Call `pipeline.Reset()` between generations. The seed is fixed by default, so set
`pipeline.Seed` before each `Reset()` if you want different output from the same
prompt.

Put the schema in the prompt as well as the mask. The mask stops the model
writing the wrong field names; it cannot tell the model which ones you wanted.

Without LLamaSharp, drive the mask yourself:

```csharp
var mask = new JsonMask(vocabulary, endOfSequence, schema);

mask.Apply(logits);       // illegal tokens become -infinity
var id = Sample(logits);  // your sampler
mask.Commit(id);          // advance the state
```

## Running the sample

Needs .NET 8 and a GGUF in `models/`. From the repo root:

```bash
# plain sampling, for comparison
dotnet run -c Release --project samples/Lattice.Smoke

# Lattice's mask, any-shape JSON
dotnet run -c Release --project samples/Lattice.Smoke -- --sampler lattice --runs 5

# Lattice's mask, constrained to a schema
dotnet run -c Release --project samples/Lattice.Smoke -- \
    --sampler lattice --schema samples/Lattice.Smoke/schemas/cat.json --runs 5
```

Each run is checked for validity and printed. With `--schema` the output is
validated against the schema independently of Lattice's own checker, so a pass
is evidence rather than a restatement.

## Schema support

Honoured: `type` (`string`, `integer`, `number`, `boolean`, `object`, `array`,
`null`), `properties`, `required`, `items`, and `enum` when every member is a
string. An undeclared `type` accepts any value, and an object with no
`properties` is free-form — checked as plain JSON until it closes.

Limits: 64 properties per object, 16 levels of declared object nesting, ASCII
property names, and no escape sequences inside a declared name.

## Layout

| path | what it is |
|---|---|
| `src/Lattice` | the library — `JsonState`, `JsonSchema`, `JsonMask`. No dependencies beyond the .NET Base Class Library. |
| `src/Lattice.LLamaSharp` | the `ISamplingPipeline` adapter. |
| `tests/Lattice.Tests` | xUnit tests. No model needed. |
| `samples/Lattice.Smoke` | console app that drives a real model. |
| `models` | GGUF files. |

## Measurements

Qwen2.5-0.5B-Instruct Q4_K_M on an Apple silicon laptop with 8 GB RAM, Release
build. 50 runs per cell, using the model's full 32k context, with each run
stopped after 2 minutes.

Valid outputs among the runs that finished:

| prompt and check | plain | lattice |
|---|---|---|
| any-shape JSON, prompted for a cat | 16/45 | 48/48 |
| `schemas/cat.json`, a flat object | 6/50 | 50/50 |
| `schemas/order.json`, nested objects and an array of objects | 5/50 | 50/50 |

In plain English: asked nicely for JSON matching the order schema, only 5 of 50
outputs were valid JSON that matched the schema. Forced through Lattice, all 50
were.

These results are for these schemas with Qwen2.5-0.5B-Instruct. Other models,
schemas and prompts will give different numbers for plain generation.
