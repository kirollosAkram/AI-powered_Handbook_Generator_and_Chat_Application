# AI Assistant - RAG-Powered Document Chat & Handbook Generator

An AI-powered application that allows you to chat with your documents and generate professional handbooks using Retrieval-Augmented Generation (RAG) with Ollama and ChromaDB.

## Features

- **💬 Document Chat**: Upload PDFs and ask questions about their content
- **📚 Handbook Generator**: Generate structured handbooks from uploaded documents
- **🔄 Chat History**: Save and manage multiple chat sessions
- **📄 PDF Export**: Export generated handbooks as professional PDFs
- **🎯 Local AI**: Runs entirely locally using Ollama models

## Prerequisites

### 1. Install Python
Python 3.9 or higher is required.

### 2. Install Ollama
Download and install Ollama from [ollama.ai](https://ollama.ai)

### 3. Download Required Models
After installing Ollama, pull the required models:

```bash
# Pull the chat/embedding models
ollama pull phi3
ollama pull nomic-embed-text