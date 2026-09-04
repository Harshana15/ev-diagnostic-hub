import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path

class PDFProcessor:
    def __init__(self, chunk_size=1000, chunk_overlap=100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # This splitter preserves semantic meaning by keeping paragraphs together
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    def process_pdf(self, pdf_path):
        print(f"Loading PDF: {pdf_path}")
        if not os.path.exists(pdf_path):
            print(f"Error: File {pdf_path} not found.")
            return []

        # 1. Load the PDF
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        
        # 2. Split into chunks
        # metadata like page number is automatically preserved by LangChain
        chunks = self.text_splitter.split_documents(pages)
        
        print(f"Split {len(pages)} pages into {len(chunks)} chunks.")
        return chunks

    def save_chunks_locally(self, chunks, output_path):
        """Optional: Save chunks to a text file for inspection."""
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, chunk in enumerate(chunks):
                f.write(f"--- Chunk {i} (Page {chunk.metadata.get('page')}) ---\n")
                f.write(chunk.page_content + "\n\n")
        print(f"Chunks saved for inspection: {output_path}")

if __name__ == "__main__":
    # Path to the manual you uploaded
    MANUAL_PATH = "BMWi3-owners-manual.pdf"
    DEBUG_OUTPUT = "data/processed/manual_chunks.txt"
    
    # Ensure directory exists
    os.makedirs("data/processed", exist_ok=True)
    
    processor = PDFProcessor(chunk_size=800, chunk_overlap=80)
    document_chunks = processor.process_pdf(MANUAL_PATH)
    
    if document_chunks:
        processor.save_chunks_locally(document_chunks, DEBUG_OUTPUT)