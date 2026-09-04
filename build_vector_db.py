import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# --------------------------------------------------
# 1. SETTINGS
# --------------------------------------------------

PDF_PATH = "BMWi3-owners-manual.pdf"
CHROMA_PATH = "data/vector_db"


# --------------------------------------------------
# 2. LOAD PDF
# --------------------------------------------------

print("Loading BMW i3 manual...")

loader = PyPDFLoader(PDF_PATH)
pages = loader.load()

print(f"Loaded {len(pages)} pages.")


# --------------------------------------------------
# 3. SPLIT INTO CHUNKS
# --------------------------------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=80
)

chunks = text_splitter.split_documents(pages)

print(f"Created {len(chunks)} chunks.")


# --------------------------------------------------
# 4. CREATE EMBEDDINGS
# --------------------------------------------------

print("Loading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded.")


# --------------------------------------------------
# 5. CREATE CHROMA DATABASE
# --------------------------------------------------

print("Creating Chroma vector database...")

os.makedirs(CHROMA_PATH, exist_ok=True)

vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=CHROMA_PATH
)

print("Vector database created successfully!")
print(f"Saved to: {CHROMA_PATH}")


# --------------------------------------------------
# 6. TEST SEARCH
# --------------------------------------------------

query = "What does the battery warning light indicate?"

results = vector_db.similarity_search(query, k=3)

print("\nTop matching chunks:\n")

for i, result in enumerate(results, start=1):
    print(f"\n--- Result {i} ---")
    print(f"Page: {result.metadata.get('page')}")
    print(result.page_content[:1000])