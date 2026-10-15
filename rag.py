from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_groq import ChatGroq
from langchain_cohere import CohereEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()


def create_qa_chain():
    """Create and return the RAG chain with hybrid search."""

    # 1. LOADING DOCUMENTS
    loader1 = TextLoader('data/SFP100.txt', encoding='utf-8')
    loader2 = TextLoader('data/SFP200.txt', encoding='utf-8')
    documents = loader1.load() + loader2.load()
    print(f"Loaded {len(documents)} documents")

    # 2. CHUNKING
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")

    # 3. EMBEDDING (via Cohere)
    embeddings = CohereEmbeddings(
        model="embed-english-light-v3.0",
        cohere_api_key=os.environ.get("COHERE_API_KEY")
    )

    # 4. VECTOR STORE
    vector_store = FAISS.from_documents(chunks, embeddings)
    print("Vector store created")

    # 5. RETRIEVER (hybrid: vector + BM25)
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 5
    vector_retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.5, 0.5]
    )

    # 6. LLM (Groq)
    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0.1,
        max_tokens=1024,  # <-- AGGIUNGI QUESTA RIGA
        api_key=os.environ.get("GROQ_API_KEY")
    )

    # 7. CHAIN
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

    return qa_chain


if __name__ == "__main__":
    qa_chain = create_qa_chain()
    questions = [
        "How much does a Premium car cost per day?",
        "What is the minimum age to rent a Premium car?",
        "What happens if I return the car without a full tank?",
        "What should I do in case of an accident?",
        "What is the phone number of the Ferrari dealership?"
    ]
    for question in questions:
        print(f"\n{'='*60}\nQUESTION: {question}\n{'='*60}")
        result = qa_chain.invoke({"query": question})
        print(f"ANSWER: {result['result']}")
        print(f"\nSOURCES:")
        for doc in result['source_documents']:
            print(f"  - {doc.metadata.get('source', 'unknown')}")