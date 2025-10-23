# ingest.py
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 1. Load documents
loader = DirectoryLoader('./knowledge_base/', glob="*.txt")
documents = loader.load()

# 2. Split documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
docs = text_splitter.split_documents(documents)

# 3. Create embeddings
model_name = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=model_name)

# 4. Create FAISS vector store and save
print("Creating vector store...")
vector_store = FAISS.from_documents(docs, embeddings)
vector_store.save_local("faiss_index")

print("Knowledge base is ready.")