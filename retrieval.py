import os
from llama_index.readers.file import PyMuPDFReader
from llama_index.core import SimpleDirectoryReader,StorageContext,Settings,VectorStoreIndex
from llama_index.core.node_parser import HierarchicalNodeParser,get_leaf_nodes
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import faiss
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever, AutoMergingRetriever
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core import load_index_from_storage
from llama_index.core.llms import MockLLM

Settings.embed_model = HuggingFaceEmbedding(model_name="mixedbread-ai/mxbai-embed-large-v1")
Settings.llm = MockLLM()

def setup(data_dir="./data", persist_dir="./storage"):
    d = 1024
    
    if os.path.exists(persist_dir):
        # Load from cache
        vector_store = FaissVectorStore.from_persist_dir(persist_dir)
        storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir=persist_dir)
        index = load_index_from_storage(storage_context)
        
        # Extract leaf nodes from docstore to rebuild BM25 (fast operation since nodes are cached)
        node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=[2048,512,128])
        all_nodes = list(storage_context.docstore.docs.values())
        leaf_nodes = get_leaf_nodes(all_nodes)
        
        bm25_retriever = BM25Retriever.from_defaults(nodes=leaf_nodes, similarity_top_k=25)
        
    else:
        # Build from scratch
        loader = PyMuPDFReader()
        documents = SimpleDirectoryReader(input_dir=data_dir, required_exts=[".pdf"],file_extractor={".pdf":loader}).load_data()

        node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=[2048,512,128])
        nodes = node_parser.get_nodes_from_documents(documents)
        leaf_nodes = get_leaf_nodes(nodes)

        faiss_index = faiss.IndexFlatL2(d)
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        storage_context.docstore.add_documents(nodes)

        index = VectorStoreIndex(leaf_nodes,storage_context=storage_context,show_progress=False)
        bm25_retriever = BM25Retriever.from_defaults(nodes=leaf_nodes, similarity_top_k=25)
        
        # Persist to cache
        os.makedirs(persist_dir, exist_ok=True)
        index.storage_context.persist(persist_dir=persist_dir)

    # A. Dense Retriever (FAISS)
    vector_retriever = index.as_retriever(similarity_top_k=25)
        
    # B. Combine them using Query Fusion
    hybrid_retriever = QueryFusionRetriever(
        [vector_retriever, bm25_retriever],
        similarity_top_k=25,
        num_queries=1,
        mode="reciprocal_rerank"
    )
    
    # C. Auto-Merging (Hierarchical)
    automerging_retriever = AutoMergingRetriever(
        hybrid_retriever,
        storage_context,
        verbose=False
    )
    
    # D. Use a local cross-encoder to rerank
    reranker = SentenceTransformerRerank(
        model="cross-encoder/ms-marco-MiniLM-L-6-v2", 
        top_n=15
    )
    
    return index, automerging_retriever, reranker, node_parser