using System.Diagnostics;
using System.Text;
using System.Text.Json;
using LLama;
using LLama.Common;
using LLama.Native;
using LLama.Sampling;

// Smoke test: load the model, generate, and check whether the output parses as JSON.
// Run from the repo root: dotnet run --project samples/Lattice.Smoke -- [--force | --grammar] [--runs N]
//   (no flag)  plain sampling
//   --force    force `{` as the first token (ForceFirstTokenPipeline)
//   --grammar  constrain with llama.cpp's built-in JSON grammar (grammars/json.gbnf)

var modelPath = args.FirstOrDefault(a => !a.StartsWith("--") && !int.TryParse(a, out _))
                ?? "models/qwen2.5-0.5b-instruct-q4_k_m.gguf";
if (!File.Exists(modelPath))
{
    Console.Error.WriteLine($"Model not found: {Path.GetFullPath(modelPath)}");
    return 1;
}

var force = args.Contains("--force");
var useGrammar = args.Contains("--grammar");
var runsIndex = Array.IndexOf(args, "--runs");
var runs = runsIndex >= 0 ? int.Parse(args[runsIndex + 1]) : 1;

// Surface only the llama.cpp log lines that show where the model landed.
NativeLibraryConfig.All.WithLogCallback((level, message) =>
{
    if (message.Contains("offloaded"))
        Console.Write($"[llama] {message}");
});

var parameters = new ModelParams(modelPath)
{
    ContextSize = 2048,
    GpuLayerCount = 99, // more than the model has, so every layer goes to the GPU
};

using var weights = LLamaWeights.LoadFromFile(parameters);
var executor = new StatelessExecutor(weights, parameters);

var template = new LLamaTemplate(weights) { AddAssistant = true };
template.Add("user", "Give me a JSON object describing a cat.");
var prompt = Encoding.UTF8.GetString(template.Apply());

var braceToken = weights.Tokenize("{", add_bos: false, special: false, Encoding.UTF8)[0];
var gbnf = useGrammar ? File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "grammars", "json.gbnf")) : null;

Console.WriteLine($"mode: {(useGrammar ? "grammar" : force ? "force" : "plain")}, runs: {runs}");

var valid = 0;
var totalPieces = 0;
var totalSeconds = 0.0;

for (var run = 1; run <= runs; run++)
{
    // A fresh pipeline per run: samplers keep per-generation state.
    ISamplingPipeline pipeline = useGrammar ? new DefaultSamplingPipeline { Grammar = new Grammar(gbnf!, "root") }
                               : force ? new ForceFirstTokenPipeline(braceToken)
                               : new DefaultSamplingPipeline();

    var inferenceParams = new InferenceParams
    {
        MaxTokens = 300, // enough room for the object to close
        AntiPrompts = ["<|im_end|>"],
        SamplingPipeline = pipeline,
    };

    var output = new StringBuilder();
    var pieces = 0;
    var clock = Stopwatch.StartNew();
    await foreach (var piece in executor.InferAsync(prompt, inferenceParams))
    {
        output.Append(piece);
        pieces++;
    }
    clock.Stop();

    var text = output.ToString().Replace("<|im_end|>", "").Trim();
    var parses = Parses(text);
    if (parses) valid++;
    totalPieces += pieces;
    totalSeconds += clock.Elapsed.TotalSeconds;

    Console.WriteLine($"\n--- run {run}: {(parses ? "VALID" : "INVALID")} JSON, {pieces} pieces in {clock.Elapsed.TotalSeconds:F2}s ---");
    Console.WriteLine(text);
}

Console.WriteLine($"\n=== {valid}/{runs} valid, {totalPieces / totalSeconds:F1} pieces/s ===");
return 0;

static bool Parses(string text)
{
    try { JsonDocument.Parse(text); return true; }
    catch (JsonException) { return false; }
}
