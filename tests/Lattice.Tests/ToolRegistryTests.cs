using Lattice.Tools;
using Xunit;

namespace Lattice.Tests;

public class ToolRegistryTests
{
    const string Parameters = "{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}}}";

    [Fact]
    public void ToolsAreKeptInTheOrderAdded()
    {
        var tools = new ToolRegistry()
            .Add("b", "second letter", Parameters)
            .Add("a", "first letter", Parameters);

        Assert.Equal(["b", "a"], tools.Tools.Select(t => t.Name));
        Assert.Equal("second letter", tools.Tools[0].Description);
        Assert.Equal(Parameters, tools.Tools[0].Parameters);
    }

    [Theory]
    [InlineData("")]
    [InlineData("has\"quote")]
    [InlineData("back\\slash")]
    [InlineData("new\nline")]
    [InlineData("café")]
    public void InvalidNamesAreRefused(string name)
    {
        Assert.Throws<ArgumentException>(() => new ToolRegistry().Add(name, "d", Parameters));
    }

    [Fact]
    public void ADuplicateNameIsRefused()
    {
        var tools = new ToolRegistry().Add("get_weather", "d", Parameters);

        Assert.Throws<ArgumentException>(() => tools.Add("get_weather", "other", Parameters));
    }

    [Theory]
    [InlineData("not json")]
    [InlineData("[]")]
    [InlineData("\"string\"")]
    public void ParametersMustBeAJsonObject(string parameters)
    {
        Assert.Throws<ArgumentException>(() => new ToolRegistry().Add("t", "d", parameters));
    }

    [Fact]
    public void MoreThanTheLimitIsRefused()
    {
        var tools = new ToolRegistry();
        for (var i = 0; i < ToolRegistry.MaxTools; i++)
            tools.Add($"t{i}", "d", Parameters);

        Assert.Throws<InvalidOperationException>(() => tools.Add("one_more", "d", Parameters));
    }
}
