import os
import chainlit as cl
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Qdrant
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Initialize Qdrant client
client = QdrantClient(location=":memory:")

# Initialize Qdrant vector store
vectorstore = Qdrant(
    client=client,
    collection_name="JohnWick",
    embeddings=embeddings
)

# Initialize ChatOpenAI
chat_model = ChatOpenAI()

# Create the RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=chat_model,
    retriever=vectorstore.as_retriever(),
    return_source_documents=True
)

@cl.on_chat_start
async def on_chat_start():
    await cl.Message(content="Welcome! Ask me anything about John Wick.").send()

@cl.on_message
async def main(message: cl.Message):
    query = message.content
    response = qa_chain({"query": query})
    
    answer = response['result']
    source_docs = response['source_documents']

    await cl.Message(content=answer).send()

    if source_docs:
        source_message = "Sources:\n"
        for i, doc in enumerate(source_docs[:3], start=1):
            source_message += f"{i}. {doc.metadata.get('source', 'Unknown')}\n"
        await cl.Message(content=source_message).send()