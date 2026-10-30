from dotenv import load_dotenv
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from file_reader import docs_path, read_multiple_pdf

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

vector_store = InMemoryVectorStore(embedding=embeddings)

docs = read_multiple_pdf(docs_path)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

all_split = text_splitter.split_documents(docs)

vector_store.add_documents(documents=all_split)

print(f"index {len(all_split)} chunk")