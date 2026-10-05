using System.Diagnostics;
using System.Text;
using System.Text.Json;
using LLama;
using LLama.Common;
using LLama.Native;
using LLama.Sampling;
using Lattice;
using Lattice.LLamaSharp;

// Smoke test: load the model, generate, and check whether the output parses as JSON.
// Run from the repo root: dotnet run --project samples/Lattice.Smoke -- [--model PATH] [--sampler MODE] [--runs N] [--schema FILE] [--prompt TEXT]
//   --model    the GGUF to load (default: models/qwen2.5-0.5b-instruct-q4_k_m.gguf)
//   --sampler  plain (default) or lattice (Lattice's JSON mask)
//   --runs     how many generations to run (default: 1)
//   --schema   constrain lattice to a JSON Schema, and put that schema in the prompt
//   --prompt   what to ask for, with no closing full stop
//              (default: "Give me a JSON object describing a cat")

var options = new Dictionary<string, string>();
for (var i = 0; i < args.Length; i++)
{
    if (args[i] is not ("--model" or "--sampler" or "--runs" or "--schema" or "--prompt"))
        return Fail($"Unknown argument: {args[i]}");
    if (i + 1 == args.Length)
        return Fail($"{args[i]} needs a value");
    options[args[i]] = args[++i];
}

var modelPath = options.GetValueOrDefault("--model", "models/qwen2.5-0.5b-instruct-q4_k_m.gguf");
if (!File.Exists(modelPath))
    return Fail($"Model not found: {Path.GetFullPath(modelPath)}");

var sampler = options.GetValueOrDefault("--sampler", "plain");
if (sampler is not ("plain" or "lattice"))
    return Fail($"Unknown sampler: {sampler} (expected plain or lattice)");
var useLattice = sampler == "lattice";

var runs = 1;
if (options.TryGetValue("--runs", out var runsText) && (!int.TryParse(runsText, out runs) || runs < 1))
    return Fail($"--runs must be a positive integer, got {runsText}");

var schemaPath = options.GetValueOrDefault("--schema");
if (schemaPath is not null && !File.Exists(schemaPath))
    return Fail($"Schema not found: {Path.GetFullPath(schemaPath)}");

var schemaText = schemaPath is null ? null : File.ReadAllText(schemaPath);
var schema = schemaText is null ? null : JsonSchema.Parse(schemaText);

// Surface only the llama.cpp log lines that show where the model landed.
NativeLibraryConfig.All.WithLogCallback((level, message) =>
{
    if (message.Contains("offloaded"))
        Console.Write($"[llama] {message}");
});

var parameters = new ModelParams(modelPath)
{
    GpuLayerCount = 99, // more than the model has, so every layer goes to the GPU
};

using var weights = LLamaWeights.LoadFromFile(parameters);

// Use the full context the model was trained for.
parameters.ContextSize = (uint)weights.ContextSize;
var executor = new StatelessExecutor(weights, parameters);

var template = new LLamaTemplate(weights) { AddAssistant = true };
var request = options.GetValueOrDefault("--prompt", "Give me a JSON object describing a cat");
template.Add("user", schemaText is null
    ? $"{request}."
    : $"{request}, matching this JSON Schema:\n{schemaText}");
var prompt = Encoding.UTF8.GetString(template.Apply());

// Leave the rest of the context for the output.
var maxTokens = weights.ContextSize - weights.Tokenize(prompt, true, true, Encoding.UTF8).Length;

// A run that has not finished by then is stopped and counted as a runaway.
var runLimit = TimeSpan.FromMinutes(2);

// Decoding the vocabulary takes a moment, so build this once and reset it per run.
using var lattice = useLattice ? new JsonSamplingPipeline(weights, schema) : null;

Console.WriteLine($"mode: {sampler}, runs: {runs}, context: {weights.ContextSize}, max tokens: {maxTokens}"
                  + (schemaPath is null ? "" : $", schema: {schemaPath}"));

var valid = 0;
var runaways = 0;
var totalPieces = 0;
var totalSeconds = 0.0;

for (var run = 1; run <= runs; run++)
{
    // Samplers keep per-generation state, so each run starts from a clean one.
    // The seed is fixed by default, so without a fresh one every run replays the
    // same draws and produces the same object.
    if (lattice is not null)
    {
        lattice.Seed = (uint)Random.Shared.Next();
        lattice.Reset();
    }

    ISamplingPipeline pipeline = useLattice ? lattice! : new DefaultSamplingPipeline();

    var inferenceParams = new InferenceParams
    {
        MaxTokens = maxTokens,
        AntiPrompts = ["<|im_end|>"],
        SamplingPipeline = pipeline,
    };

    var output = new StringBuilder();
    var pieces = 0;
    bool runaway;
    using var limit = new CancellationTokenSource(runLimit);
    var clock = Stopwatch.StartNew();
    try
    {
        await foreach (var piece in executor.InferAsync(prompt, inferenceParams, limit.Token))
        {
            output.Append(piece);
            pieces++;
        }
    }
    catch (OperationCanceledException)
    {
    }
    clock.Stop();

    // The executor can stop quietly on cancellation instead of throwing.
    runaway = limit.IsCancellationRequested;

    var text = output.ToString().Replace("<|im_end|>", "").Trim();
    var violation = runaway ? $"runaway, stopped after {runLimit.TotalMinutes:F0} min"
                  : schemaText is null ? Parses(text) ? null : "does not parse"
                  : SchemaViolation(text, schemaText);
    if (violation is null) valid++;
    if (runaway) runaways++;
    totalPieces += pieces;
    totalSeconds += clock.Elapsed.TotalSeconds;

    Console.WriteLine($"\n--- run {run}: {(violation is null ? "VALID" : "INVALID: " + violation)}, {pieces} pieces in {clock.Elapsed.TotalSeconds:F2}s ---");
    Console.WriteLine(text);
}

Console.WriteLine($"\n=== {valid}/{runs} valid, {runaways} runaway, {totalPieces / totalSeconds:F1} pieces/s ===");
return 0;

static int Fail(string message)
{
    Console.Error.WriteLine(message);
    return 1;
}

static bool Parses(string text)
{
    try { JsonDocument.Parse(text); return true; }
    catch (JsonException) { return false; }
}

static string? SchemaViolation(string text, string schemaText)
{
    JsonDocument document;
    try { document = JsonDocument.Parse(text); }
    catch (JsonException e) { return $"does not parse ({e.Message})"; }

    using var _ = document;
    var results = Json.Schema.JsonSchema.FromText(schemaText).Evaluate(
        document.RootElement, new Json.Schema.EvaluationOptions { OutputFormat = Json.Schema.OutputFormat.List });
    if (results.IsValid) return null;

    var failure = results.Details?.LastOrDefault(d => d.Errors is { Count: > 0 });
    return failure is null ? "does not match the schema"
                           : $"{failure.InstanceLocation}: {failure.Errors!.Values.First()}";
}
