# AI Research Paper Analysis

[![Demo Video](https://drive.google.com/thumbnail?id=1nS46CFyGL7uqQarQUCsVhOmJI4sKCdRh&sz=w1200-h630)](https://drive.google.com/file/d/1nS46CFyGL7uqQarQUCsVhOmJI4sKCdRh/view?usp=sharing)
*(Click the banner above to watch the demo video)*

## Overview
This project is an AI-powered Research Paper Analysis application built using Langchain. It allows users to upload research papers (PDFs), automatically extracts and refines sections from the document, provides detailed summaries for specific topics/sections, and includes an interactive chat feature powered by RAG (Retrieval-Augmented Generation) to ask questions about the paper's contents.

## Pipeline Architecture
The complete pipeline involves several stages, combining classical NLP with modern Large Language Models.

### 1. Document Upload & Processing Pipeline
*   **PDF Upload:** The user uploads a research paper in PDF format via the web interface.
*   **Text Extraction:** Extracts the raw text from the uploaded PDF document.
*   **Section Extraction:** Identifies the major headings and sections within the raw text.
*   **Section Refinement (LLM):** Leverages a Groq-powered LLM to clean, standardize, and accurately identify the main section headings.
*   **Content Splitting:** Maps the refined headings back to the original text to extract the specific content corresponding to each section.

### 2. Summarization Pipeline
*   **Topic Selection:** The user selects a specific section/topic extracted from the paper.
*   **Detailed Summarization (LLM):** Uses the LLM to generate a comprehensive, easy-to-understand summary specifically focused on the content of the chosen section.

### 3. RAG (Retrieval-Augmented Generation) Chat Pipeline
*   **Vector Database Creation:** 
    *   The entire text of the document is split into smaller, overlapping chunks.
    *   These chunks are converted into dense vector embeddings using **HuggingFace Embeddings**.
    *   A **FAISS** vector database is initialized and populated with these embeddings for fast similarity search.
*   **Retrieval & QA:** 
    *   When the user asks a question, the FAISS database retrieves the most relevant text chunks related to the query.
    *   The LLM (Groq) uses this retrieved context to generate a precise, grounded answer to the user's question, minimizing hallucinations.

## Technologies Used
*   **Backend Framework:** Flask (Python)
*   **LLM Orchestration:** LangChain
*   **Language Model Provider:** Groq
*   **Embedding Model:** HuggingFace Embeddings
*   **Vector Store:** FAISS (Facebook AI Similarity Search)
