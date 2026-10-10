using System.Diagnostics;
using System.Text;
using System.Text.Json;
using LLama;
using LLama.Common;
using LLama.Native;
using LLama.Sampling;
using Lattice;
using Lattice.LLamaSharp;
using Lattice.Tools;
using Lattice.Tools.Formats;

// Smoke test: load the model, generate, and check whether the output parses as JSON.
// Run from the repo root: dotnet run --project samples/Lattice.Smoke -- [--model PATH] [--sampler MODE] [--runs N] [--schema FILE] [--prompt TEXT] [--tools DIR] [--prompts FILE]
//   --model    the GGUF to load (default: models/qwen2.5-0.5b-instruct-q4_k_m.gguf)
//   --sampler  plain (default) or lattice (Lattice's JSON mask)
//   --runs     how many generations to run per prompt (default: 1)
//   --schema   constrain lattice to a JSON Schema, and put that schema in the prompt
//   --prompt   what to ask for, with no closing full stop
//              (default: "Give me a JSON object describing a cat")
//   --tools    a folder of tool definitions, one JSON file each with name, description and
//              parameters; the model must answer with a call to one of them
//   --prompts  with --tools, a JSON array of { "prompt", "tool" } pairs, each run --runs times
//              and checked for the expected tool

var options = new Dictionary<string, string>();
for (var i = 0; i < args.Length; i++)
{
    if (args[i] is not ("--model" or "--sampler" or "--runs" or "--schema" or "--prompt" or "--tools" or "--prompts"))
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

var toolsPath = options.GetValueOrDefault("--tools");
if (toolsPath is not null && !Directory.Exists(toolsPath))
    return Fail($"Tools folder not found: {Path.GetFullPath(toolsPath)}");
if (toolsPath is not null && schemaPath is not null)
    return Fail("--tools and --schema cannot be combined");

var promptsPath = options.GetValueOrDefault("--prompts");
if (promptsPath is not null && toolsPath is null)
    return Fail("--prompts needs --tools");
if (promptsPath is not null && options.ContainsKey("--prompt"))
    return Fail("--prompt and --prompts cannot be combined");
if (promptsPath is not null && !File.Exists(promptsPath))
    return Fail($"Prompts not found: {Path.GetFullPath(promptsPath)}");

var schemaText = schemaPath is null ? null : File.ReadAllText(schemaPath);

// Tool mode: the registry supplies both the prompt and the schema.
var format = new QwenFormat();
ToolRegistry? tools = null;
if (toolsPath is not null)
{
    tools = new ToolRegistry();
    foreach (var file in Directory.GetFiles(toolsPath, "*.json").Order())
    {
        using var definition = JsonDocument.Parse(File.ReadAllText(file));
        var root = definition.RootElement;
        tools.Add(root.GetProperty("name").GetString()!,
                  root.GetProperty("description").GetString()!,
                  root.GetProperty("parameters").GetRawText());
    }

    schemaText = format.Schema(tools);
}

var schema = schemaText is null ? null : JsonSchema.Parse(schemaText);

// Each case is a request and, with --prompts, the tool it should call.
var defaultRequest = tools is null ? "Give me a JSON object describing a cat" : "What's the weather in Paris?";
List<(string Request, string? Expected)> cases = promptsPath is null
    ? [(options.GetValueOrDefault("--prompt", defaultRequest), null)]
    : JsonDocument.Parse(File.ReadAllText(promptsPath)).RootElement.EnumerateArray()
        .Select(c => (c.GetProperty("prompt").GetString()!, (string?)c.GetProperty("tool").GetString()))
        .ToList();

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

// A run that has not finished by then is stopped and counted as a runaway.
var runLimit = TimeSpan.FromMinutes(2);

// Decoding the vocabulary takes a moment, so build this once and reset it per run.
using var lattice = useLattice ? new JsonSamplingPipeline(weights, schema) : null;

Console.WriteLine($"mode: {sampler}, runs: {runs}, context: {weights.ContextSize}"
                  + (schemaPath is null ? "" : $", schema: {schemaPath}")
                  + (toolsPath is null ? "" : $", tools: {string.Join(", ", tools!.Tools.Select(t => t.Name))}")
                  + (cases.Count > 1 ? $", prompts: {cases.Count}" : ""));

var valid = 0;
var correct = 0;
var invented = 0;
var runaways = 0;
var totalPieces = 0;
var totalSeconds = 0.0;
var perCase = new List<(string Request, int Valid, int Correct)>();

foreach (var (request, expected) in cases)
{
    var prompt = tools is not null ? format.Prompt(tools, request) : ChatPrompt(request);

    // Leave the rest of the context for the output.
    var maxTokens = weights.ContextSize - weights.Tokenize(prompt, true, true, Encoding.UTF8).Length;

    if (cases.Count > 1)
        Console.WriteLine($"\n##### {request}" + (expected is null ? "" : $"  (expected: {expected})"));

    var caseValid = 0;
    var caseCorrect = 0;

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
        string? called = null;
        string? violation;
        if (runaway)
            violation = $"runaway, stopped after {runLimit.TotalMinutes:F0} min";
        else if (tools is not null)
        {
            var call = format.Parse(text);
            var tool = tools.Tools.FirstOrDefault(t => t.Name == call?.Name);
            var whole = SchemaViolation(CallText(text), schemaText!);

            if (call is null)
                violation = whole ?? "not a tool call";
            else if (tool is null)
            {
                violation = $"invented tool \"{call.Name}\"";
                invented++;
            }
            else if (whole is null)
            {
                violation = null;
                called = call.Name;
            }
            else
                // Name the called tool's own failure rather than the union's.
                violation = SchemaViolation(call.Arguments, tool.Parameters) is { } arguments
                    ? $"{call.Name} arguments: {arguments}"
                    : whole;
        }
        else
            violation = schemaText is null ? Parses(text) ? null : "does not parse"
                      : SchemaViolation(text, schemaText);

        var isCorrect = violation is null && (expected is null || called == expected);
        if (violation is null) { valid++; caseValid++; }
        if (isCorrect) { correct++; caseCorrect++; }
        if (runaway) runaways++;
        totalPieces += pieces;
        totalSeconds += clock.Elapsed.TotalSeconds;

        var verdict = violation is not null ? "INVALID: " + violation
                    : called is null ? "VALID"
                    : isCorrect ? $"VALID, called {called}"
                    : $"VALID, WRONG TOOL: called {called}";
        Console.WriteLine($"\n--- run {run}: {verdict}, {pieces} pieces in {clock.Elapsed.TotalSeconds:F2}s ---");
        Console.WriteLine(text);
    }

    perCase.Add((request, caseValid, caseCorrect));
}

var total = cases.Count * runs;
if (cases.Count > 1)
{
    Console.WriteLine("\n=== per prompt: valid, correct tool ===");
    foreach (var (request, caseValid, caseCorrect) in perCase)
        Console.WriteLine($"{caseValid}/{runs}  {caseCorrect}/{runs}  {request}");
}

Console.WriteLine($"\n=== {valid}/{total} valid"
                  + (cases.Any(c => c.Expected is not null) ? $", {correct}/{total} correct tool" : "")
                  + (tools is null ? "" : $", {invented} invented tool names")
                  + $", {runaways} runaway, {totalPieces / totalSeconds:F1} pieces/s ===");
return 0;

// The prompt for plain JSON mode, with the schema in it when there is one.
string ChatPrompt(string request)
{
    var template = new LLamaTemplate(weights) { AddAssistant = true };
    template.Add("user", schemaText is null
        ? $"{request}."
        : $"{request}, matching this JSON Schema:\n{schemaText}");
    return Encoding.UTF8.GetString(template.Apply());
}

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

// The call's JSON from a tool-mode reply: the text before any closing </tool_call>.
static string CallText(string text)
{
    var close = text.IndexOf("</tool_call>", StringComparison.Ordinal);
    return (close >= 0 ? text[..close] : text).Trim();
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
