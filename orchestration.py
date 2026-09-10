import os
import logging
import warnings

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.disable(logging.WARNING)

from typing import TypedDict, List 
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from retrieval import setup
from dotenv import load_dotenv
from llama_index.core import QueryBundle, Settings
import json
import faiss
import numpy as np

# Semantic Cache Globals
CACHE_FAISS_PATH = "storage/query_cache.faiss"
CACHE_JSON_PATH = "storage/query_answers.json"
query_faiss_index = None
query_answers = {}

def load_semantic_cache():
    global query_faiss_index, query_answers
    os.makedirs("storage", exist_ok=True)
    if os.path.exists(CACHE_FAISS_PATH):
        query_faiss_index = faiss.read_index(CACHE_FAISS_PATH)
        if os.path.exists(CACHE_JSON_PATH):
            with open(CACHE_JSON_PATH, "r") as f:
                data = json.load(f)
                query_answers = {int(k): v for k, v in data.items()}
    else:
        query_faiss_index = faiss.IndexIDMap(faiss.IndexFlatL2(1024))
        query_answers = {}

def save_semantic_cache():
    global query_faiss_index, query_answers
    faiss.write_index(query_faiss_index, CACHE_FAISS_PATH)
    with open(CACHE_JSON_PATH, "w") as f:
        json.dump(query_answers, f)

# Load API keys from keys.env
load_dotenv("keys.env")
class RAGState(TypedDict, total=False):
    question: str
    optimized_query: str 
    context_chunks : List[str]
    final_answer : str
    cache_hit: bool
    query_embedding: List[float]

def semantic_cache_node(state: RAGState):
    global query_faiss_index, query_answers
    if query_faiss_index is None:
        load_semantic_cache()
        
    print("\n[Worker 0] Interceptor: Checking Semantic Cache for past questions...")
    question = state["question"]
    emb = Settings.embed_model.get_query_embedding(question)
    
    if query_faiss_index.ntotal > 0:
        emb_np = np.array([emb], dtype=np.float32)
        distances, indices = query_faiss_index.search(emb_np, 1)
        best_dist = distances[0][0]
        
        # A relaxed threshold of 0.5 (around -75% semantic similarity)
        if indices[0][0] != -1 and best_dist < 0.5:
            print(f"         -> Semantic Cache HIT! (L2 Distance: {best_dist:.4f})")
            answer = query_answers.get(indices[0][0], "Cache corrupted.")
            return {"cache_hit": True, "query_embedding": emb, "final_answer": answer}
        else:
            print(f"         -> Cache Miss. (Closest past question was L2 Distance: {best_dist:.4f} - Threshold is 0.5)")
            return {"cache_hit": False, "query_embedding": emb}
            
    print("         -> Cache Miss. (Cache is empty). Routing to pipeline...")
    return {"cache_hit": False, "query_embedding": emb}

def query_processor_node(state : RAGState):
    print("\n[Worker 1] Query Processor: Optimizing user question into a formal database query...")
    question = state["question"] 

    llm = ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPEN_ROUTER"),
        model="openai/gpt-4o",
        max_tokens=500
    )
    sys_prompt = """ You are an expert legal search optimizer. Your job is to take the user's conversational
    question and rewrite it into a highly effective search query for a Vector Database.
    Remove conversational filler. Expand acronyms if obvious. Use formal legal terminology if appropriate. 
    OUTPUT ONLY THE OPTIMIZED SEARCH QUERY. Do not include quotes or explanations."""
    messages = [SystemMessage(content = sys_prompt),HumanMessage(content=question)]

    response = llm.invoke(messages)
    optimized_query = response.content.strip()
    return {"optimized_query" : optimized_query}


def retrieve_node(state: RAGState):
    print("[Worker 2] Researcher: Connecting to FAISS Vector Database and retrieving exact chunks...")

    query_to_search = state["optimized_query"]

    # mute stdout and stderr
    import sys
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')
    
    try:
        index, hybrid_retriever, reranker, node_parser = setup()
        retrieved_nodes = hybrid_retriever.retrieve(query_to_search)
        top_n_nodes = reranker.postprocess_nodes(retrieved_nodes, query_bundle=QueryBundle(query_to_search))
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        
    formatted_context = []
    for node in top_n_nodes:
        filename = node.metadata.get('file_name', 'Unknown')
        page = node.metadata.get('source', 'Unknown')
        formatted_context.append(f"[Source: {filename}, Page: {page}]\n{node.text}")
        
    return {"context_chunks": formatted_context}

def synthesize_node(state: RAGState):
    print("[Worker 3] Synthesizer: Reviewing retrieved legal context and writing final answer...")

    question = state["question"] 
    context_text = "\n\n".join(state["context_chunks"])

    llm = ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPEN_ROUTER"),
        model="anthropic/claude-sonnet-4.6",
        max_tokens=1500
    )
    
    sys_prompt = f"""You are a brilliant legal assistant.
    Answer the user's question using ONLY the provided context below.
    If the context does not contain the answer, say "I do not know based on the provided documents."
    
    CRITICAL RULE: At the end of every factual sentence you write, you MUST 
    include an inline citation using the Source and Page number provided in the context.
    Example: "The plaintiff argued for damages [Source: comp26286.pdf, Page: 4]."
    
    CONTEXT:
    {context_text}
    """
    
    messages = [SystemMessage(content=sys_prompt),HumanMessage(content=question)]
    
    response = llm.invoke(messages)
    final_answer = response.content
    
    # Save to cache
    global query_faiss_index, query_answers
    if query_faiss_index is None:
        load_semantic_cache()
        
    emb = state.get("query_embedding")
    if emb:
        new_id = query_faiss_index.ntotal
        emb_np = np.array([emb], dtype=np.float32)
        query_faiss_index.add_with_ids(emb_np, np.array([new_id]))
        query_answers[new_id] = final_answer
        save_semantic_cache()
        
    return {"final_answer": final_answer}

def should_continue(state: RAGState):
    if state.get("cache_hit"):
        return END
    return "QueryProcessor"

def build_graph():
    workflow = StateGraph(RAGState)
    
    workflow.add_node("SemanticCache", semantic_cache_node)
    workflow.add_node("QueryProcessor", query_processor_node)
    workflow.add_node("Researcher", retrieve_node)
    workflow.add_node("Synthesizer", synthesize_node)
    
    workflow.add_edge(START, "SemanticCache")
    workflow.add_conditional_edges("SemanticCache", should_continue)
    workflow.add_edge("QueryProcessor", "Researcher")
    workflow.add_edge("Researcher", "Synthesizer")
    workflow.add_edge("Synthesizer", END)
    
    return workflow.compile()

if __name__ == "__main__":
    app = build_graph()
    
    # Give the graph an initial state
    initial_state = {"question": "What misconducts was Chiueh involved in?"}
    
    # Run the graph!
    print("Starting LangGraph Pipeline...")
    final_state = app.invoke(initial_state)
    
    print("\n\n=== FINAL ANSWER ===")
    print(final_state["final_answer"])