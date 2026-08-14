# RAG-Based Chatbot for Malaysia Labour Law

A **Retrieval-Augmented Generation (RAG)** chatbot designed to answer questions about **Malaysia Labour Law** using Large Language Models (LLMs), semantic search, vector databases, and reranking.

The system retrieves relevant legal information from a curated knowledge base before generating an answer, helping improve factual grounding and reduce hallucinations commonly associated with standalone LLMs.

> **Disclaimer:** This project is for educational and research purposes only. It does not constitute legal advice. Always refer to the latest official Malaysian legislation or consult a qualified legal professional for legal matters.

---

## 📌 Project Overview

Understanding Malaysian labour law can be challenging due to the amount of legislation, amendments, regulations, and legal terminology involved.

This project explores how **Retrieval-Augmented Generation (RAG)** can be used to build a domain-specific chatbot capable of answering questions related to Malaysian employment and labour regulations.

## Data & Sources
Source: https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://jtksm.mohr.gov.my/sites/default/files/2023-11/Akta%2520Kerja%25201955%2520%2528Akta%2520265%2529.pdf&ved=2ahUKEwj0r9vjmZ-WAxXsSGcHHZH3HVYQFnoECCEQAQ&usg=AOvVaw1FwFOWifmvi059_gcNjyEh

## Features
- Focused on Malaysia Labour Law
- Semantic document retrieval using vector embeddings
- Vector database for efficient similarity search
- Reranking to improve retrieval relevance
- Reduced hallucination through context-grounded generation
- Potential support for source/citation display

## Technology Stack

| Category | Technology | Purpose |
|---|---|---|
| Programming Language | Python | Core development and RAG pipeline |
| LLM | Large Language Model | Generate natural-language answers |
| RAG Framework | LangChain | Build and manage the RAG pipeline |
| Embedding Model | Sentence Transformers | Convert legal documents and queries into vector embeddings |
| Vector Database | FAISS | Store and perform similarity search on document embeddings |
| Reranker | Cross-Encoder | Re-rank retrieved legal documents based on query relevance |
| Document Processing | pdfplumber | Extract text from Malaysian labour law PDF document |
| Text Splitting | Recursive Character Text Splitter | Split legal documents into smaller searchable chunks |
| Frontend | Streamlit | Provide an interactive chatbot interface |

## Project Structure
```
Labour Law Chatbot
│
├── Akta Kerja 1955(Akta 265).pdf    # Malaysia Labour Law PDF
├── Labour Law.md                    # Knowledge base of RAG
├── OCR                              # Text extraction
├── chatbot.py                       # Main file to run chatbot
└── requirements.txt                 # Python dependencies
```
