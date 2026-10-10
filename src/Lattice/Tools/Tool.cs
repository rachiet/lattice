namespace Lattice.Tools;

/// A function the model may call.
///
/// `Parameters` is the JSON Schema for the call's arguments, as text. It is
/// shown to the model in the prompt and enforced on the arguments it writes.
public sealed record Tool(string Name, string Description, string Parameters);
