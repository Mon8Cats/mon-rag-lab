# RAG

- Simple prompting
- Vector DBs + Cosine Similarity
- Specialized AI Agents
- Just throw everything into the context window


## Steps

- chroma db clinet
- get or create collection
- upload pdf
  - open file
  - pdf reader
  - each page -> extract
    - text, id -> add to collection
- get query input
  - find closest pages
- create response using ollama chat
  - model = llama3.1
  - message = {role, content=closetpages}

## LLM Knowledge

- Pre-training
- Fine Tuning
- In Context Learning
- question -> smart lookup, relevant documents
  - question + relevant documents 
  - LLM -> generate answer
- raw data source -> info extraction -> chunking -> embedding -> vector database -> 
- query -> embedding -> vector database
- relevant data -> llms -> response


## Steps

- Input query
- Chunking: source documents are preprocessed into smaller chunks
- Embedding Creation: chunks are converted into embeddings
- Storage in Vector Database: embeddings are stored in a vector database for efficient retrieval
- Retrieval: the system retrieves the most relevant chunks based on the query (embedding the query and do a similarity search within the vector database)
- Prompt Construction: the query and retrieved chunks are combined into a prompt.
- Generation: the language model generates using the prompt as context.

## Other techniques

- Reranker: rank again
- Hybrid search: vector search, keyword search
- Agentic RAG: query translation/planning (abstraction, reasoning)
  - corrective RAG agent