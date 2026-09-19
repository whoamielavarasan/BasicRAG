# query.py
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
assert os.getenv("GROQ_API_KEY"), "Please set your GROQ_API_KEY inside your .env configuration file."

DB_DIR = "./chroma_db"

# 1. Connect to the Existing Disk-Based Vector Store
print("Connecting to local disk vector storage...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = Chroma(
    collection_name="telecom_collection",
    embedding_function=embeddings,
    persist_directory=DB_DIR
)

# 2. Configure the Retriever to fetch the top 3 matches
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# 3. Setup Fast ChatGroq Engine
llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.2)

# 4. System Prompt Enforcements
system_prompt = (
    "You are an expert technical assistant. Answer the user's question using ONLY "
    "the following pieces of retrieved context from the technical document stored on disk. "
    "If you cannot find the answer in the context, state clearly that it is missing.\n\n"
    "Context:\n{context}"
)
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# 5. Build the Production RAG Pipeline Chain
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# 6. Execution Loop Interactivity
print("\n Vector Search Engine Online. Type 'exit' to quit.")
while True:
    user_query = input("\nAsk your PDF a question: ")
    if user_query.strip().lower() == "exit":
        print("Goodbye!")
        break
        
    if not user_query.strip():
        continue
        
    print("Searching vector boundaries and generating answer via Groq...")
    response = rag_chain.invoke({"input": user_query})
    
    print("\n[ANSWER]:")
    print(response["answer"])
