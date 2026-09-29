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
// Run from the repo root: dotnet run --project samples/Lattice.Smoke -- [--grammar | --lattice] [--runs N] [--schema FILE]
//   (no flag)  plain sampling
//   --grammar  constrain with llama.cpp's built-in JSON grammar (grammars/json.gbnf)
//   --lattice  constrain with Lattice's JSON mask
//   --schema   constrain --lattice to a JSON Schema, and put that schema in the prompt

// Read before the model path, which otherwise claims the first bare argument.
var schemaIndex = Array.IndexOf(args, "--schema");
var schemaPath = schemaIndex >= 0 ? args[schemaIndex + 1] : null;
if (schemaPath is not null && !File.Exists(schemaPath))
{
    Console.Error.WriteLine($"Schema not found: {Path.GetFullPath(schemaPath)}");
    return 1;
}

var modelPath = args.FirstOrDefault(a => !a.StartsWith("--") && a != schemaPath && !int.TryParse(a, out _))
                ?? "models/qwen2.5-0.5b-instruct-q4_k_m.gguf";
if (!File.Exists(modelPath))
{
    Console.Error.WriteLine($"Model not found: {Path.GetFullPath(modelPath)}");
    return 1;
}

var useGrammar = args.Contains("--grammar");
var useLattice = args.Contains("--lattice");
var runsIndex = Array.IndexOf(args, "--runs");
var runs = runsIndex >= 0 ? int.Parse(args[runsIndex + 1]) : 1;

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
    ContextSize = 2048,
    GpuLayerCount = 99, // more than the model has, so every layer goes to the GPU
};

using var weights = LLamaWeights.LoadFromFile(parameters);
var executor = new StatelessExecutor(weights, parameters);

var template = new LLamaTemplate(weights) { AddAssistant = true };
template.Add("user", schemaText is null
    ? "Give me a JSON object describing a cat."
    : $"Give me a JSON object describing a cat, matching this JSON Schema:\n{schemaText}");
var prompt = Encoding.UTF8.GetString(template.Apply());

var gbnf = useGrammar ? File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "grammars", "json.gbnf")) : null;

// Decoding the vocabulary takes a moment, so build this once and reset it per run.
using var lattice = useLattice ? new JsonSamplingPipeline(weights, schema) : null;

Console.WriteLine($"mode: {(useLattice ? "lattice" : useGrammar ? "grammar" : "plain")}, runs: {runs}"
                  + (schemaPath is null ? "" : $", schema: {schemaPath}"));

var valid = 0;
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

    ISamplingPipeline pipeline = useLattice ? lattice!
                               : useGrammar ? new DefaultSamplingPipeline { Grammar = new Grammar(gbnf!, "root") }
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
    var violation = schemaText is null
        ? Parses(text) ? null : "does not parse"
        : SchemaViolation(text, schemaText);
    if (violation is null) valid++;
    totalPieces += pieces;
    totalSeconds += clock.Elapsed.TotalSeconds;

    Console.WriteLine($"\n--- run {run}: {(violation is null ? "VALID" : "INVALID: " + violation)}, {pieces} pieces in {clock.Elapsed.TotalSeconds:F2}s ---");
    Console.WriteLine(text);
}

Console.WriteLine($"\n=== {valid}/{runs} valid, {totalPieces / totalSeconds:F1} pieces/s ===");
return 0;

static bool Parses(string text)
{
    try { JsonDocument.Parse(text); return true; }
    catch (JsonException) { return false; }
}

// Checks the output against the schema using only System.Text.Json, so a pass is
// evidence about the mask rather than the mask agreeing with itself. Returns the
// first violation found, or null when the object matches.
static string? SchemaViolation(string text, string schemaText)
{
    JsonDocument parsed;
    try { parsed = JsonDocument.Parse(text); }
    catch (JsonException e) { return $"does not parse ({e.Message})"; }

    using var document = parsed;
    using var schemaDocument = JsonDocument.Parse(schemaText);

    var root = document.RootElement;
    var schema = schemaDocument.RootElement;
    var properties = schema.GetProperty("properties");

    if (root.ValueKind != JsonValueKind.Object) return $"root is {root.ValueKind}, not an object";

    foreach (var written in root.EnumerateObject())
    {
        if (!properties.TryGetProperty(written.Name, out var declared))
            return $"undeclared property \"{written.Name}\"";

        if (TypeMismatch(declared, written.Value) is { } mismatch)
            return $"\"{written.Name}\": {mismatch}";

        if (declared.TryGetProperty("enum", out var members)
            && written.Value.ValueKind == JsonValueKind.String
            && !members.EnumerateArray().Any(m => m.GetString() == written.Value.GetString()))
            return $"\"{written.Name}\": \"{written.Value.GetString()}\" is not an enum member";

        if (declared.TryGetProperty("items", out var items) && written.Value.ValueKind == JsonValueKind.Array)
            foreach (var item in written.Value.EnumerateArray())
                if (TypeMismatch(items, item) is { } itemMismatch)
                    return $"\"{written.Name}\" item: {itemMismatch}";
    }

    if (schema.TryGetProperty("required", out var required))
        foreach (var name in required.EnumerateArray())
            if (!root.TryGetProperty(name.GetString()!, out _))
                return $"missing required \"{name.GetString()}\"";

    return null;
}

static string? TypeMismatch(JsonElement declared, JsonElement value)
{
    if (!declared.TryGetProperty("type", out var type) || type.GetString() is not { } expected)
        return null;

    var matches = expected switch
    {
        "string" => value.ValueKind == JsonValueKind.String,
        "integer" => value.ValueKind == JsonValueKind.Number && value.TryGetInt64(out _),
        "number" => value.ValueKind == JsonValueKind.Number,
        "boolean" => value.ValueKind is JsonValueKind.True or JsonValueKind.False,
        "object" => value.ValueKind == JsonValueKind.Object,
        "array" => value.ValueKind == JsonValueKind.Array,
        "null" => value.ValueKind == JsonValueKind.Null,
        _ => true,
    };

    return matches ? null : $"{value.ValueKind} is not {expected}";
}
