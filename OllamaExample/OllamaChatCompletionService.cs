using System.Threading.Tasks;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;
using OllamaSharp;

public class OllamaChatCompletionService : IChatCompletionService
{
    private readonly IOllamaApiClient _ollamaApiClient;

    public OllamaChatCompletionService(IOllamaApiClient ollamaApiClient)
    {
        _ollamaApiClient = ollamaApiClient;
    }

    public IReadOnlyDictionary<string, object?> Attributes => new Dictionary<string, object?>();

    public async Task<IReadOnlyList<ChatMessageContent>> GetChatMessageContentsAsync(
        ChatHistory chatHistory, 
        PromptExecutionSettings? executionSettings = null, 
        Kernel? kernel = null, 
        CancellationToken cancellationToken = default)
    {
        var request = CreateChatRequest(chatHistory);

        var response = await OllamaApiClient.Chat(request, cancellationToken);
        return 
        [
            new ChatMessageContent
            {
                Role = GetAuthorRole(response.Message.Role) ?? AuthorRole.Assistant,
                Content = response.Message.Content,
                InnerContent = response,
                ModelId = "llama3.1"
            }
        ];
    }

    public async IAsyncEnumerable<StreamingChatMessageContent> GetStreamingChatMessageContentsAsync(
        ChatHistory chatHistory, 
        PromptExecutionSettings? executionSettings = null, 
        Kernel? kernel = null, 
        CancellationToken cancellationToken = default)
    {
        var request = CreateChatRequest(chatHistory);

        await foreach(var response in _ollamaApiClient.StreamChat(request, cancellationToken))
        {
            yield return new StreamingChatMessageContent();
        }
    }
}
