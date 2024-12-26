# NET and Chat System

- Building the Basic Chat Interface
- Integrating the LLM
- Handling User Input


## OLLAM Chat Completion

### Parameters

- model
- messages : to keep chat memory
  - message:
    - role: system, user, assistant, tool
    - content: the content of message
    - images:a list of images
    - tool_calls: a list of tools the model wants to use
- tools: model uses tools





### Steps

- Embedding:
- Microsoft.KernelMemory, 
  - KernelMemory.FileSystem.DevTools
  - KernelMemory.MemoryStorage.DevTools
- SimpleChatSystem
- ollamEndpoint
- ollamClient
- selectModel
- ImportDocument()
  - DocumentImporter()
    - Import()
  - WebPageImporter()
    - Import()
- GetMemoryKernel()
  - KernelMemoryBuilder()
    - WithCustomPromptPRovider()
    - WithCustomEmbeddingGenerator()
    - WithCustomTextGenerator()
    - WithSimpleVectorDb()
    - Build<MemoryServerless>();
    - return memoryBuilder.Build<MemoryServerless>();
    - use this meory for what?



### C# Library for AI

- Microsoft.Extensions.AI.Abstractions
- IChatClient
- Microsoft.SemanticKernel
- Azure OpenAI SDK
- OpenAI SDK
- Microsoft.SemanticKernel.Connectors
  - Postgres, SqlServer, Chroma, DuckDB, Milvus, MongoDB, Pinecone
- Microsoft.Extensions.AI.Abstractions
- IChatClient client = new OllamaChatClient(...);
- var response = await chatClient.CompleteAsync(...);
- response.Message;
- app.Services.AddChatClient(builder => builder.Use...);
- dotnet add pacakge Microsoft.SemanticKernel
- var builder = Kernel.CreateBuilder();
- var kernel = builder.Build();var builder = WebApplication.CreateBuilder();
- builder.Services.AddKernel();
- var app = builder.Build();