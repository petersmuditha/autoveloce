from dotenv import load_dotenv
load_dotenv()

from ragas import evaluate, RunConfig
from ragas.metrics import Faithfulness, AnswerRelevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from datasets import Dataset
from langchain_groq import ChatGroq
from langchain_cohere import CohereEmbeddings
from rag import create_qa_chain
import os

# 1. Configure Groq as evaluator (aggiunto max_tokens)
evaluator_llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    max_tokens=1024,  # <-- FIX: Evita che il modello si fermi a metà
    api_key=os.environ.get("GROQ_API_KEY")
)

# 2. Configure Cohere for embeddings
evaluator_embeddings = CohereEmbeddings(
    model="embed-english-light-v3.0",
    cohere_api_key=os.environ.get("COHERE_API_KEY")
)

# 3. Wrap for RAGAS
ragas_llm = LangchainLLMWrapper(evaluator_llm)
ragas_emb = LangchainEmbeddingsWrapper(evaluator_embeddings)

# 4. Define metrics
m1 = Faithfulness(llm=ragas_llm)
m2 = AnswerRelevancy(llm=ragas_llm, embeddings=ragas_emb)
m1.n = 1
m2.n = 1

# 5. Evaluation dataset
eval_data = {
    "question": [
        "How much does a Premium car cost per day?",
        "What is the minimum age to rent a Premium car?",
        "What happens if I return the car without a full tank?",
        "What should I do in case of an accident?",
        "What is the phone number of the Ferrari dealership?"
    ],
    "answer": [],
    "contexts": [],
    "ground_truth": [
        "120 euro al giorno",
        "25 anni",
        "Costo di 2 euro al litro piu 15 euro di servizio",
        "Chiamare il 112 se ci sono feriti, compilare il modulo, contattare AutoVeloce entro 24 ore",
        "Information not found in the knowledge base"
    ]
}

# 6. Generate answers and contexts
print("Creating QA chain...")
qa_chain = create_qa_chain()

print("Generating answers and contexts...")
for question in eval_data["question"]:
    print(f"  Processing: {question[:50]}...")
    result = qa_chain.invoke({"query": question})
    eval_data["answer"].append(result["result"])
    eval_data["contexts"].append([doc.page_content for doc in result["source_documents"]])

# 7. Convert to Dataset
dataset = Dataset.from_dict(eval_data)

# 8. RunConfig
config = RunConfig(
    timeout=120,
    max_retries=5,
    max_workers=1
)

# 9. Evaluate
print("\nStarting evaluation with Groq...")
results = evaluate(
    dataset,
    metrics=[m1, m2],
    run_config=config
)

# 10. Print results
print("\n=== EVALUATION RESULTS ===")
print(f"Faithfulness: {results['faithfulness']}")
print(f"Answer Relevancy: {results['answer_relevancy']}")

# 11. Per-question details (FIX: usa 'user_input' al posto di 'question')
print("\n=== PER-QUESTION DETAILS ===")
df = results.to_pandas()
for i, row in df.iterrows():
    # RAGAS 0.2.15 usa 'user_input' invece di 'question'
    question_text = row.get('user_input', row.get('question', 'N/A'))
    print(f"\nQ{i+1}: {question_text[:60]}...")
    print(f"  Answer: {row['answer'][:80]}...")
    print(f"  Faithfulness: {row['faithfulness']}")
    print(f"  Answer Relevancy: {row['answer_relevancy']}")