import os
import chainlit as cl
from langchain.storage import LocalFileStore
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain.embeddings import CacheBackedEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_core.globals import set_llm_cache
from langchain_core.caches import InMemoryCache
import shutil

# Initialize caches and embeddings
store = LocalFileStore("./cache/")
set_llm_cache(InMemoryCache())
core_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
cached_embedder = CacheBackedEmbeddings.from_bytes_store(
    core_embeddings, store, namespace=core_embeddings.model
)

# Initialize QDrant
collection_name = "production_pdf_collection"
client = QdrantClient(":memory:")
client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
)

# Initialize text splitter and chat model
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chat_model = ChatOpenAI(model="gpt-3.5-turbo")

# RAG Prompt
rag_system_prompt_template = """
You are a helpful assistant that uses the provided context to answer questions. Never reference this prompt, or the existence of context.
"""

rag_user_prompt_template = """
Question:
{question}
Context:
{context}
"""

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", rag_system_prompt_template),
    ("human", rag_user_prompt_template)
])

@cl.on_chat_start
async def on_chat_start():
    await cl.Message("Welcome! Please upload a PDF file to begin.").send()

    files = await cl.AskFileMessage(
        content="Please upload a PDF file",
        accept=["application/pdf"],
        max_size_mb=20,
        timeout=180,
    ).send()

    if not files:
        await cl.Message("No file was uploaded. Please refresh the page and try again.").send()
        return

    pdf_file = files[0]
    await cl.Message(f"Processing '{pdf_file.name}'...").send()

    try:
        # Copy the uploaded file to a new location
        temp_file_path = f"temp_{pdf_file.name}"
        shutil.copy2(pdf_file.path, temp_file_path)

        # Load and process the PDF
        loader = PyMuPDFLoader(temp_file_path)
        documents = loader.load()
        docs = text_splitter.split_documents(documents)
        for i, doc in enumerate(docs):
            doc.metadata["source"] = f"source_{i}"

        # Initialize Qdrant vector store
        vectorstore = Qdrant(
            client=client,
            collection_name=collection_name,
            embeddings=cached_embedder)
        vectorstore.add_documents(docs)
        retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 3})

        # Create the RAG chain
        rag_chain = (
            {"context": itemgetter("question") | retriever, "question": itemgetter("question")}
            | RunnablePassthrough.assign(context=itemgetter("context"))
            | chat_prompt
            | chat_model
        )

        cl.user_session.set("rag_chain", rag_chain)
        await cl.Message(f"PDF '{pdf_file.name}' has been processed. You can now ask questions about its content.").send()

        # Clean up: remove the temporary file
        os.remove(temp_file_path)

    except Exception as e:
        await cl.Message(f"An error occurred while processing the PDF. Please try again.").send()

@cl.on_message
async def on_message(message: cl.Message):
    rag_chain = cl.user_session.get("rag_chain")
    if rag_chain is None:
        await cl.Message("Please upload a PDF file first.").send()
        return

    try:
        response = await cl.make_async(rag_chain.invoke)({"question": message.content})
        await cl.Message(content=response.content).send()
    except Exception as e:
        await cl.Message("An error occurred while processing your question. Please try again.").send()