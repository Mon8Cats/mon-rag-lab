# RAG and Ollama 

check these videos: https://www.youtube.com/@decoder-sh

## 1. Set Environment

```bash
python -m venv .venv
source .venv/bin/activate
```

## 2. Install Dependencies

```bash
python -m pip install ollama numpy 
```

## 3. Check Ollama Functions

```bash
import ollama
ollama.chat(model="mistral', messages=[
    {'role':'user', 'content': 'Why is the sky blue?'}
])
=> response
{
    'model': 'mistral',
    'created_at': '2024-12-27T17:47:18.7933792',
    'message': 
        {
            'role': 'assistant',
            'content': "..."}
        },
    'done': True,
    'total_duration': 9781194708,
    'load_duration': 7847796458,
    'prompt_eval_count': 15, 
    'prompt_eval_duration': 9669500,
    'eval_count': 100,
    'eval_duration': 1835609000
}

ollama.embeddings(model='mistral', prompt='The dog ran fast')
=>
```

## 4. Prepare Data

- Books in Project Gutenberg, download
- Parse the documents
- Embeddings

```bash
import ollama
import time
import os
import json

# main.py
def parse_file(filename):
    with open(filename, encoding="utf-8-sig") as f:
        paragraphs = []
        buffer = []
        for line in f.readlines():
            line = line.strip()
            if line:
                buffer.append(line)
            elif len(buffer):
                paragraphs.append((" ").join(buffer))
                buffer = []
        if len(buffer):
            paragraphs.append((" ").join(buffer))
        return paragraphs

def save_embeddings(filename, embeddings):
    # create dir if it doesn't exist
    if not os.path.exists("embeddings"):
        os.makedirs("embeddings")
    # dump embeddings to json
    with open(f"embeddings/{filename}.json", "w") as f:
        json.dump(embeddings, f)

def load_embeddings(filename):
    # check if file exists
    if not os.path.exists(f"embeddings/{filename}.json"):
        return False
    # load embeddings from json
    with open(f"embeddings/{filename}.json", "r") as f:
        return json.load(f)


# old
"""
def get_embeddings(modelname, chunks):
    return [
        ollama.embeddings(model=modelname, prompt=chunk)["embedding"]
        for chunk in chunks
    ]
"""

# new
def get_embeddings(filename, modelname, chunks):
    # check if embeddings are already saved
    if (embeddings := load_embeddings(filename)) is not False:
        return embeddings

    # get embeddings from ollama
    embeddings = [
        ollama.embeddings(model=modelname, prompt=chunk)["embedding"]
        for chunk in chunks
    ]

    # save embeddings
    save_embeddings(filename, embeddings)
    return embeddings

def find_most_similar(needle, haystack):
    needle_norm = norm(needle)
    similarity_score = [
        np.dot(needle, item) / (needle_norm * norm (item)) for item in haystack
    ]
    return sorted(zip(similarity_scores, range(len(haystack))), reverse=True)

def main():
    SYSTEM_PROMPT = """You are a helpful reading assistant who answers questions
        based on snippets of text provided in context. Answer only using the context provided, 
        being as concise as possible. If you're unsure, just say that you don't know.
        Context:
    """


    # open file
    filename = "peter-pan.txt"
    paragraphs = parse_file(filename)

    # join multiple paragraphs to meet a minimum length (chunking function?)
    # use different model

    #start = time.perf_counter()
    #embeddings = get_embeddings('mistral', paragraphs[5:90])
    embeddings = get_embeddings(filename, "mistral", paragraphs)
    #print(time.pert_counter() - start)
    #print(paragraphs[:10])
    #print(len(embeddings))

    #prompt = "who is the story's primary villain?"
    prompt = input("What do you want to know? -> ")
    prompt_embedding = ollama.embeddings(model="mistral", prompt=prompt)[
        "embedding"
    ]
    # find most similar embeddings?
    most_similar_chunks = find_most_similar(prompt_embedding, embeddings)[:5]
    #for item in most_similar_chunks:
    #    print(item[0], paragraphs[item[1]]) # similarity score, paragraph

    response = ollama.chat(
        model="mistral",
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
                + "\n".join(paragraphs[item[1]]) for item in most_similar_chunks),
            },
            {"role": "user", "content": prompt},

        ],
    )

    print("\n\n")
    print(response["message"]["content"])



if __name__ == "__main__":
    main()

```

## Use Local Model?

- open source models are good
- cheaper
- privacy (no external APIs)
- not depends on external connectivity
- code is not important but the reasoning behind is important

## Steps

1. download ollama 
   1. a model = a gigantic mathematical formula
   2. ollama = the common wrapper (interface) around all of these different models
   3. ollama site = a list of models
   4. > ollama -> list commands
   5. > ollama pull [model-name]
   6. > ollama list -> list downloaded models
   7. download model means download files (.ollama/)
   8. ollama run [model-name]
   9. >tell me a joke
   10. > /bye or ctrl + d
   11. create a fold to developer a rag system using Langchain to get data from PDF files
   12. >mkdir local-model
   13. >open vs-code
   14. >add a jupyter plug in, python plug in, 
   15. >create a notebook in the folder notebook.ipynb
   16. >create a venv
   17. local-model> python3 -m venv .venv # m = module
   18. >source .venv/bin/activate
   19. open the notebook> add a code in a cell > it asks to choose a kernel> choose the python env created (.venv). 
   20. then it install ipykernel 
   21. create an env file (.env file)
   22. 




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
