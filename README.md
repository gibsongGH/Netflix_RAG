---
title: Netflix_RAG
emoji: 🎬
colorFrom: red
colorTo: purple
sdk: gradio
sdk_version: "5.22.0"
python_version: "3.10"
app_file: app.py
pinned: false
---

# Netflix Culture RAG Chatbot

A demo chatbot built with **Retrieval-Augmented Generation (RAG)** that answers questions about the [Netflix Culture Memo](https://jobs.netflix.com/culture), 
[Work-Life Philosophy](https://jobs.netflix.com/work-life-philosophy) and [2025 Annual Report Intro](https://s22.q4cdn.com/959853165/files/doc_financials/2025/ar/99482238-46b2-4d0d-b292-40e6781bdf03.pdf).

## How it works

1. The documents are chunked and embedded using OpenAI's `text-embedding-3-small` model at startup.
2. Embeddings are stored in an in-memory [ChromaDB](https://www.trychroma.com/) vector database.
3. When you ask a question, the most relevant chunks are retrieved and passed as context to `gpt-4.1-mini`.
4. The response is streamed back via a [Gradio](https://gradio.app/) chat interface.

## Tech stack

- **LLM**: OpenAI `gpt-4.1-mini`
- **Embeddings**: OpenAI `text-embedding-3-small`
- **Vector DB**: ChromaDB (in-memory)
- **UI**: Gradio

## Running locally

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=your_key_here
python app.py
```

## Live demo

Try it on [Hugging Face Spaces](https://huggingface.co/spaces/gibsongHF/Netflix_RAG).
