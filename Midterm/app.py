from fastapi import FastAPI
from pydantic import BaseModel
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Qdrant
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv(".env.local")

# Initialize FastAPI app
app = FastAPI()

# Initialize OpenAI and embeddings
llm = OpenAI(temperature=0)
embeddings = OpenAIEmbeddings()

# Load your vector store (adjust as needed)
qdrant = Qdrant.load("path/to/your/qdrant")

# Create the RAG chain
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=qdrant.as_retriever(),
    return_source_documents=True
)

class Query(BaseModel):
    question: str

@app.post("/query")
async def query(query: Query):
    result = rag_chain({"query": query.question})
    return {
        "answer": result["result"],
        "sources": [doc.page_content for doc in result["source_documents"]]
    }