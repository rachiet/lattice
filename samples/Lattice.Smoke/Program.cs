using System.Text;
using LLama;
using LLama.Common;
using LLama.Native;
using LLama.Sampling;

// Smoke test: load the model, confirm Metal is used, generate a few tokens.
// Run from the repo root: dotnet run --project samples/Lattice.Smoke

var modelPath = args.FirstOrDefault(a => !a.StartsWith("--")) ?? "models/qwen2.5-0.5b-instruct-q4_k_m.gguf";
if (!File.Exists(modelPath))
{
    Console.Error.WriteLine($"Model not found: {Path.GetFullPath(modelPath)}");
    return 1;
}

// Surface only the llama.cpp log lines that show where the model landed.
NativeLibraryConfig.All.WithLogCallback((level, message) =>
{
    if (message.Contains("offloaded") || message.Contains("ggml_metal_init: found device"))
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

// Pass --force to make `{` the only legal first token.
var force = args.Contains("--force");
var braceTokens = weights.Tokenize("{", add_bos: false, special: false, System.Text.Encoding.UTF8);
if (force)
    Console.WriteLine($"Forcing first token: id {braceTokens[0]} (\"{{\" tokenizes to {braceTokens.Length} token(s))");

var inferenceParams = new InferenceParams
{
    MaxTokens = 60,
    AntiPrompts = ["<|im_end|>"],
    SamplingPipeline = force ? new ForceFirstTokenPipeline(braceTokens[0]) : new DefaultSamplingPipeline(),
};

Console.WriteLine("\n--- output ---");
var tokens = 0;
var clock = System.Diagnostics.Stopwatch.StartNew();
await foreach (var piece in executor.InferAsync(prompt, inferenceParams))
{
    Console.Write(piece);
    tokens++;
}
clock.Stop();

Console.WriteLine($"\n--- {tokens} pieces in {clock.Elapsed.TotalSeconds:F2}s ---");
return 0;
