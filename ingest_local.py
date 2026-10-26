import sys
sys.stdout.reconfigure(encoding="utf-8")
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

print("Đang đọc file tai_lieu_xe.txt...")
loader = TextLoader("tai_lieu_xe.txt", encoding="utf-8")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
splits = text_splitter.split_documents(docs)

print("Đang số hóa tài liệu vào ChromaDB Local...")
embeddings = OllamaEmbeddings(model="nomic-embed-text", num_gpu=0)
vectorstore = Chroma(
    collection_name="showroom_xe",
    persist_directory="./chroma_db_local", 
    embedding_function=embeddings
)
try:
    vectorstore.delete_collection()
except Exception:
    pass

vectorstore = Chroma.from_documents(
    documents=splits, 
    embedding=embeddings, 
    collection_name="showroom_xe",
    persist_directory="./chroma_db_local"
)

print("✅ Hoàn tất! AI đã học xong tài liệu xe của bạn.")