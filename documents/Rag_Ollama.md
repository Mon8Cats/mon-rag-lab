# Ollama 

## Implement RAG with Lamma using Ollama and Langchain

1. Install Ollama

```bash
    # install ollama, download llama
    ollama run llama3.1
```

2. Set up the environment

```bash
    # download libraries
    pip install langchain langchain_community langchain-openai scikit-learn langchain-ollama
```

3. Load and prepare documents

```bash
    from langchain_community.document_loaders import WebBaseLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    # List of URLs to load documents from
    urls = [
        "<https://lilianweng.github.io/posts/2023-06-23-agent/>",
        "<https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/>",
        "<https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/>",
    ]
    # Load documents from the URLs
    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]
```

4. Split documents into chunks

```bash
# Initialize a text splitter with specified chunk size and overlap
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0
)
# Split the documents into chunks
doc_splits = text_splitter.split_documents(docs_list)
```

5. Create a vector store

```bash
    # text chunks -> embeddings -> store in a vector store 
    # for quick and efficient retrieval based on similarity

    from langchain_community.vectorstores import SKLearnVectorStore
    from langchain_openai import OpenAIEmbeddings
    # Create embeddings for documents and store them in a vector store
    vectorstore = SKLearnVectorStore.from_documents(
        documents=doc_splits,
        embedding=OpenAIEmbeddings(openai_api_key="api_key"),
    )
    retriever = vectorstore.as_retriever(k=4)

```

6. Set up the LLM and prompt template

```bash
    from langchain_ollama import ChatOllama
    from langchain.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    # Define the prompt template for the LLM
    prompt = PromptTemplate(
        template="""You are an assistant for question-answering tasks.
        Use the following documents to answer the question.
        If you don't know the answer, just say that you don't know.
        Use three sentences maximum and keep the answer concise:
        Question: {question}
        Documents: {documents}
        Answer:
        """,
        input_variables=["question", "documents"],
    )

    # Initialize the LLM with Llama 3.1 model
    llm = ChatOllama(
        model="llama3.1",
        temperature=0, # more deterministic and less random outputs
    )

    # Create a chain combining the prompt template and LLM
    rag_chain = prompt | llm | StrOutputParser()
```

7. Integrate the retriever and LLM into a RAG application

```bash
    # Define the RAG application class
    class RAGApplication:
        def __init__(self, retriever, rag_chain):
            self.retriever = retriever
            self.rag_chain = rag_chain
        def run(self, question):
            # Retrieve relevant documents
            documents = self.retriever.invoke(question)
            # Extract content from retrieved documents
            doc_texts = "\\n".join([doc.page_content for doc in documents])
            # Get the answer from the language model
            answer = self.rag_chain.invoke({"question": question, "documents": doc_texts})
            return answer
```

8. Test the application

```bash
    # Initialize the RAG application
    rag_application = RAGApplication(retriever, rag_chain)
    # Example usage
    question = "What is prompt engineering"
    answer = rag_application.run(question)
    print("Question:", question)
    print("Answer:", answer)

```
