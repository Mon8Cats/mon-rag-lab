using System.Text.Json.Serialization;
using OllamaSharp.Models;

public class ModelsResponse
{
    [JsonPropertyName("models")]
    public List<Model> Models {get; set;}
}