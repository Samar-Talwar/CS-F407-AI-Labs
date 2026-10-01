# CS F407 Lab, Week 7 | Author: Samar Talwar | Not licensed for reuse or submission by others.

"""
Week 7: Learning Autoregressive Models: Transformers, Ollama & RAG.

This package provides:
1. PyTorch Transformer blocks (Scaled Dot-Product Attention, Multi-Head Attention,
   Positional Encoding, Causal Masking, Decoder-Only AR Model).
2. Ollama Client wrapper with offline deterministic stub and real HTTP backend.
3. RAG pipeline with word-level chunking, BM25/TF-IDF retrieval, prompt assembly,
   grounded generation, and evaluation metrics (Recall@K, MRR).
"""

from .ollama_client import OllamaChain, OllamaClient, PromptTemplate
from .rag import (
    BM25Retriever,
    DocumentChunk,
    RAGPipeline,
    TFIDFRetriever,
    evaluate_rag,
    extract_text_chunks,
    mean_reciprocal_rank,
    recall_at_k,
)
from .transformer import (
    DecoderOnlyTransformer,
    FeedForward,
    MultiHeadAttention,
    PositionalEncoding,
    create_causal_mask,
    generate_text,
    scaled_dot_product_attention,
    train_autoregressive_model,
)

__all__ = [
    "scaled_dot_product_attention",
    "create_causal_mask",
    "PositionalEncoding",
    "MultiHeadAttention",
    "FeedForward",
    "DecoderOnlyTransformer",
    "train_autoregressive_model",
    "generate_text",
    "OllamaClient",
    "PromptTemplate",
    "OllamaChain",
    "DocumentChunk",
    "extract_text_chunks",
    "TFIDFRetriever",
    "BM25Retriever",
    "RAGPipeline",
    "recall_at_k",
    "mean_reciprocal_rank",
    "evaluate_rag",
]
