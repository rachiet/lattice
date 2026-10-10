namespace Lattice.Tools;

/// A call read from model output: the tool's name and its arguments as JSON text.
public sealed record ToolCall(string Name, string Arguments);
