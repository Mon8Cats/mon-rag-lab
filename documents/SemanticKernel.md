# Semantic Kernel

- Semantic Kernel is a lightweight, open-source development kit that lets me easily build AI agents and integrate the latest AI models into my C#, Python, or Java codebase.




## Semantic Kernel

- RAG: the process of providing additional context when prompting a LLM.
- RAG involves retrieving additional data that is relevant to the current user ask and augmenting the prompt sent to the LLM with the data.
- LLM can use its training plus the additional context to provide a more accurate response.
- Allow the AI model to extract the search query or queries from the user ask and use function calling to retrieve the relevant information it needs.
- Function calling with Bing text search
- Function calling with Bing text search and citations
- Function calling with Bing text search and filtering

## Vector Stores

- Use the Vector Store connector to retrieve the record collection I want to search.
- Wrap the record collection with VectorStoreTextSearch.
- Convert to plugin for use in RAG and/or function calling scenarios.


## Steps

- Download Ollama
- ollama pull mistral // a model
- ollam run mistral 
- send a message

## Steps

- Setup the environment
- Bring text into the database
- Find the relevant document chunks form the database
- Populate a prompt and send it to a model to generate an answer.

### Setup the environment

- the vector store:
  - Chroma? Docker container?
    - docker run -d -p 8000:8000 -v chroma-data:/chromadb/data
- https://www.youtube.com/watch?v=9KEUFe4KQAI&list=PLvsHpqLkpw0fIT-WbjY-xBRxTftjwiTLB


### Using CLI

- ollama run [model]
- ollama help
- ollama serve
- OLLAMA_KEEP_ALIVE=0 ollama server # set an env variable
- 
- Commands:
  - server, crete, show, run, pull, push, list, ps, cp, m, help
  - create: model? # modelfile?
    - FROM llama3.1
    - SYSTEM """ // multiline text
      - You are the ...
      - """
  - ollama create swede -f ./modelfile  # create new model
  - ollama run swede # my custom model name
- From Huggingface Safetensors 
- hfdownloader -s . -m ClosedCharacter/Peach-9B-8k-Roleplay
- # use hfdownloader tool
  - FROM .
  - SYSTEM """
  -  some text
  -  """
  - TEMPLATE """
     -  from where?
  - """
  - PARAMETER stop <|im_end|> # stopword
  - ollama create peach -f ./modelfile -q Q$_0
  - # transferring model data 
  - ollama run peach
  - # create a model from safetensors
  - ollama show peach # show highlevel of the model?
  - ollama show -h?
  - ollama run llama3.1 --format json
  - ollama run llama3.1 --keepalive 20
  - ollama run llama3.1 --verbose 
  - ollama pull nuextract # for api, pull first?
  - ollama ls or list # list models
  - ollama cp phi3 phi3x (copy only reference?)
  - ollama rm 
  - ollama push # push a new model to ollama.com?