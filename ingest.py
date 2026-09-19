# ingest.py
import os
from dotenv import load_dotenv
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.indexing import index

load_dotenv()

# 1. Setup paths
DB_DIR = "./chroma_db"
CACHE_DIR = "./index_cache_base"
PDF_PATH = "Elavarasan_Azure_.Net_Engineer.pdf"

print("Initializing local embedding engine...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 2. Initialize Persistent Chroma DB Instance
vector_store = Chroma(
    collection_name="telecom_collection",
    embedding_function=embeddings,
    persist_directory=DB_DIR
)

# 3. Initialize the Record Manager (Tracks modifications/hashes)
from langchain_classic.indexes import SQLRecordManager

os.makedirs(CACHE_DIR, exist_ok=True)
namespace = "chroma/telecom_collection"
record_manager = SQLRecordManager(
    namespace=namespace,
    db_url=f"sqlite:///{CACHE_DIR}/record_manager.sql"
)
record_manager.create_schema()

# 4. Load and Chunk Document
print(f"Loading document: {PDF_PATH}")
loader = PyMuPDF4LLMLoader(file_path=PDF_PATH)
pages = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
chunks = text_splitter.split_documents(pages)

# 5. Run the Smart Indexing Update Command
print("Syncing documents with local Chroma DB storage...")
indexing_result = index(
    chunks,
    record_manager,
    vector_store,
    cleanup="full", # Automatically purges old deleted chunk records from disk
    source_id_key="source"
)

print("\n--- Ingestion Sync Report ---")
print(f"Num Added:   {indexing_result['num_added']}")
print(f"Num Updated: {indexing_result['num_updated']}")
print(f"Num Skipped: {indexing_result['num_skipped']}")
print(f"Num Deleted: {indexing_result['num_deleted']}")
print("Vector Store update complete and saved safely to disk.")
