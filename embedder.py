import torch
from sentence_transformers import SentenceTransformer
from typing import List

class DocumentEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        # Automatically uses GPU if available, else CPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Initializing Embedder on device: {self.device}")
        self.model = SentenceTransformer(model_name, device=self.device)

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Converts a list of text strings into a list of vector embeddings.
        """
        print(f"Generating embeddings for {len(texts)} chunks...")
        # convert_to_tensor=False returns numpy arrays, which ChromaDB/FAISS prefer
        embeddings = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        """
        Converts a single user question into a vector for searching the database.
        """
        return self.model.encode(query).tolist()

if __name__ == "__main__":
    # Mock data for testing the script standalone
    test_chunks = [
        "The high-voltage battery is located in the underbody of the vehicle.",
        "In case of an accident, the high-voltage system is automatically switched off."
    ]
    
    embedder = DocumentEmbedder()
    vectors = embedder.generate_embeddings(test_chunks)
    
    print(f"Generated {len(vectors)} vectors.")
    print(f"Vector dimension: {len(vectors[0])}") # Should be 384 for MiniLM