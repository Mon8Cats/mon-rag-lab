var ollamaEndpoint = "http://127.0.0.1:11434";
var ollamaClient = new HttpClinet
{
    BaseAddress = new Uri(ollamaEndpoint)
};


var responseMessage = await ollamaClient.GetAsync("/api/tags");
var content = await responseMessage.Content.REadAsStringAsync();
Console.WriteLine(content);
