import chromadb
import os


class VectorDatabase:

    def __init__(self, db_path="./data/vector_db"):
        self.db_path = db_path

        os.makedirs(self.db_path, exist_ok=True)

        self.client = chromadb.PersistentClient(path=self.db_path)

        self.collection = self.client.get_or_create_collection(
            name="ev_manuals",
            metadata={"hnsw:space": "cosine"}
        )

    def add_documents(self, chunks, embeddings):

        print(f"Adding {len(chunks)} chunks to Vector Store...")
        print(f"DEBUG: chunks = {len(chunks)}")
        print(f"DEBUG: embeddings = {len(embeddings)}")

        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Chunks ({len(chunks)}) and embeddings ({len(embeddings)}) "
                "must have the same length."
            )

        if len(embeddings) == 0:
            raise ValueError("Embeddings list is EMPTY.")

        ids = [f"id_{i}" for i in range(len(chunks))]

        documents = [
            chunk.page_content
            for chunk in chunks
        ]

        metadatas = [
            chunk.metadata
            for chunk in chunks
        ]

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )

        print("Vector Store populated successfully.")

    def query(self, query_embedding, n_results=5):

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )

        return results


if __name__ == "__main__":

    from pdf_chunker import PDFProcessor
    from embedder import DocumentEmbedder

    MANUAL_PATH = "BMWi3-owners-manual.pdf"

    print(f"Loading PDF: {MANUAL_PATH}")

    processor = PDFProcessor()
    chunks = processor.process_pdf(MANUAL_PATH)

    embedder = DocumentEmbedder()

    embeddings = embedder.generate_embeddings(
        [chunk.page_content for chunk in chunks]
    )

    print(f"DEBUG: chunks = {len(chunks)}")
    print(f"DEBUG: embeddings = {len(embeddings)}")

    vdb = VectorDatabase(
        db_path="./data/vector_db"
    )

    vdb.add_documents(
        chunks,
        embeddings
    )

    test_queries = [
        "Where is the high-voltage battery located?",
        "What happens to the high-voltage system in an accident?",
        "How do I charge the BMW i3?",
        "What does the blue charging indicator mean?",
        "What does the battery warning light mean?"
    ]

    for test_query in test_queries:

        print("\n" + "=" * 70)
        print(f"QUESTION: {test_query}")

        query_vec = embedder.embed_query(test_query)

        search_results = vdb.query(
            query_vec,
            n_results=5
        )

        print("\nTOP RETRIEVED CHUNKS:")

        for i, document in enumerate(
            search_results["documents"][0]
        ):

            print("\n" + "-" * 70)
            print(f"RESULT {i + 1}")

            print(
                f"Distance: "
                f"{search_results['distances'][0][i]}"
            )

            print(
                f"Metadata: "
                f"{search_results['metadatas'][0][i]}"
            )

            print("\nDocument:")
            print(document)