using System.Text;
using LLama;
using LLama.Common;
using LLama.Native;

// Smoke test: load the model, confirm Metal is used, generate a few tokens.
// Run from the repo root: dotnet run --project samples/Lattice.Smoke

var modelPath = args.Length > 0 ? args[0] : "models/qwen2.5-0.5b-instruct-q4_k_m.gguf";
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

var inferenceParams = new InferenceParams
{
    MaxTokens = 60,
    AntiPrompts = ["<|im_end|>"],
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
