# Lattice — constrained decoding engine for .NET

## What it is

A library that makes schema-invalid LLM output impossible, not just unlikely.

At each generation step the model emits a logit for every vocabulary token
(~151k for Qwen2.5). Before a token is sampled, Lattice sets the logit of every
token that would violate the grammar (JSON Schema first, arbitrary grammars
later) to negative infinity. The model keeps freedom over *what* it says and
loses all freedom over *shape*. Output parses and matches the schema on the
first try — no parse-and-retry loop.

No training or fine-tuning. The model is frozen; all work happens at inference
time between "logits produced" and "token chosen". This requires local weights —
hosted APIs return finished text, so there are no logits to mask.

## Requirements

- Runs fully locally; no API keys or network calls.
- Targets modest hardware: must work on a machine with 8GB RAM and no discrete
  GPU. Use small quantized models and a context size of 2048 by default.
- C# / .NET 8.

## Stack

- `LLamaSharp` 0.27.0 + `LLamaSharp.Backend.Cpu` 0.27.0 (includes the Metal
  binaries for osx-arm64; there is no separate Metal package). No need to build llama.cpp.
- Reference model: `qwen2.5-0.5b-instruct-q4_k_m.gguf`, placed in `models/`
  (gitignored). GGUF contains weights, tokenizer vocab and metadata.
- Hook: `LLama.Sampling.ISamplingPipeline`. `BaseSamplingPipeline.ProcessLogits`
  (older) or `Apply(ctx, LLamaTokenDataArray)` (newer) exposes mutable logits.
  Setting `InferenceParams.SamplingPipeline` overrides all other sampling.

## Layout

- `src/Lattice` — the library
- `samples/Lattice.Smoke` — console app: load model, generate, force tokens
- `tests/Lattice.Tests`
- `benchmarks` — plain vs constrained runs
- `models` — GGUF files (gitignored)

## Roadmap

1. Smoke test: load GGUF, confirm GPU backend is active, generate a few tokens.
2. Custom `ISamplingPipeline` that forces a single specific token.
3. JSON Schema → state machine compiler.
4. Prefix index (trie) over the token vocabulary, read from the GGUF tokenizer.
5. Fast per-step legal-token lookup. Naive = scan all vocab tokens per step.
   Real version = cached legal-token bitset masks per parser state.
6. Benchmarks: same model and tasks, plain vs constrained; count parse failures
   and schema violations. Repeat across 0.5B / 1.5B / 3B.

Hard part throughout: tokens don't align to grammar boundaries — a single token
may be `":` or `,"` or `"}\n`.

## Status

Steps 1–2 done. `samples/Lattice.Smoke` loads the model with all layers on Metal
(~40 tokens/s); `--force` uses `ForceFirstTokenPipeline` to force `{` (token 90)
as the first token. Next: roadmap step 3.

Gotcha: `StatelessExecutor` does not call `ISamplingPipeline.Accept`, so state
that must advance per token cannot rely on it.
