# AI Ethics RAG System

This project implements a Retrieval-Augmented Generation (RAG) system focused on AI ethics, using documents from the White House's Blueprint for an AI Bill of Rights and NIST's AI Risk Management Framework.

## Features

- Document loading and preprocessing using PyMuPDF and LangChain
- Text splitting for optimal chunk size
- Embeddings generation using OpenAI's models
- Vector storage with Qdrant
- Question-answering capability using a RAG chain
- Test set generation using Ragas for evaluation

## Setup

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env.local` file with your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   LANGCHAIN_API_KEY=your_langchain_api_key_here
   ```

## Usage

Run the Jupyter notebook `midterm_assignment.ipynb` to:
1. Load and preprocess documents
2. Generate embeddings and store in Qdrant
3. Set up the RAG chain
4. Generate test sets for evaluation

## API Deployment

The RAG system is deployed as an API using FastAPI and can be accessed via Hugging Face Spaces.

To query the model:
