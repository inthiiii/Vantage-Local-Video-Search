import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # <--- The Magic Fix

import faiss
import numpy as np
import faiss
import numpy as np
import os
import pickle
from typing import List, Tuple
from src.embedding_engine import EmbeddingEngine

class VectorDB:
    """
    Manages the storage and retrieval of vector embeddings using FAISS.
    """
    
    def __init__(self, index_path: str = "data/vector_index.bin", metadata_path: str = "data/metadata.pkl"):
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.engine = EmbeddingEngine()
        self.metadata = [] # Stores file paths corresponding to vector IDs
        self.index = None
        
        # Load existing index if available
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            print("📂 Loading existing Vector DB...")
            self.index = faiss.read_index(self.index_path)
            with open(self.metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
        else:
            print("🆕 Creating new Vector DB...")
            # 512 is the dimension size of CLIP (ViT-B/32)
            self.index = faiss.IndexFlatIP(512) 

    def build_index(self, frames_folder: str):
        """
        Scans the folder, converts all images to vectors, and adds them to FAISS.
        """
        image_files = [f for f in os.listdir(frames_folder) if f.endswith(('.jpg', '.png'))]
        
        if not image_files:
            print("⚠️ No images found to index.")
            return

        print(f"🚀 Indexing {len(image_files)} frames. This might take a moment...")
        
        vectors = []
        valid_files = []
        
        for i, filename in enumerate(image_files):
            path = os.path.join(frames_folder, filename)
            
            # Get vector from our M3-optimized engine
            vector = self.engine.get_image_embedding(path)
            
            if vector is not None:
                vectors.append(vector)
                valid_files.append(path)
                
            # Progress print every 10 frames
            if (i + 1) % 10 == 0:
                print(f"   Processed {i + 1}/{len(image_files)}")

        # Convert list to numpy array for FAISS
        if vectors:
            vectors_np = np.array(vectors).astype('float32')
            
            # Add to FAISS index
            self.index.add(vectors_np)
            self.metadata.extend(valid_files)
            
            # Save to disk
            faiss.write_index(self.index, self.index_path)
            with open(self.metadata_path, 'wb') as f:
                pickle.dump(self.metadata, f)
            
            print(f"✅ Indexing complete! Saved {len(vectors)} vectors.")
        else:
            print("❌ No valid vectors generated.")

    def search(self, query: str, k: int = 3) -> List[str]:
        """
        Searches the database for the top 'k' images matching the text query.
        """
        print(f"🔍 Searching for: '{query}'")
        
        # 1. Convert text to vector
        query_vector = self.engine.get_text_embedding(query)
        query_np = np.array([query_vector]).astype('float32')
        
        # 2. Search FAISS
        # D is distances (similarity scores), I is indices (IDs)
        D, I = self.index.search(query_np, k)
        
        results = []
        for idx in I[0]:
            if idx < len(self.metadata):
                results.append(self.metadata[idx])
        
        return results

# --- Test Block ---
if __name__ == "__main__":
    # 1. Initialize DB
    db = VectorDB()
    
    # 2. Build Index (Point this to your frames folder from Phase 1)
    # Only run this once! Comment it out after the first run if you want to just test search.
    db.build_index("data/frames")
    
    # 3. Test Search
    # Change "person" to something that actually exists in your test video
    results = db.search("person", k=2) 
    print(f"\n🎉 Top Matches: {results}")
