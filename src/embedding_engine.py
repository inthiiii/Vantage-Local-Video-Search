import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from typing import List
import os

class EmbeddingEngine:
    """
    The 'Brain' of Vantage. Converts images and text into vector embeddings 
    using the CLIP model.
    """
    
    def __init__(self):
        # M3 Specific Optimization: Use Metal Performance Shaders (MPS) if available
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"⚡ Loading Vision Engine on: {self.device.upper()}")
        
        # We use 'openai/clip-vit-base-patch32' - it's efficient and accurate
        self.model_id = "openai/clip-vit-base-patch32"
        
        try:
            self.processor = CLIPProcessor.from_pretrained(self.model_id)
            self.model = CLIPModel.from_pretrained(self.model_id).to(self.device)
            print("✅ Model loaded successfully.")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise e

    def get_image_embedding(self, image_path: str):
        """
        Reads an image and returns its vector embedding.
        """
        try:
            image = Image.open(image_path)
            
            # Preprocess the image (resize, normalize) and move to M3 GPU
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            
            # Generate embedding (no_grad because we are inferencing, not training)
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
            
            # Normalize the vector (crucial for accurate search later)
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            
            # Move back to CPU and convert to list (for storage)
            return image_features.cpu().detach().numpy().flatten()
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None

    def get_text_embedding(self, text: str):
        """
        Converts a search query (e.g., "red car") into a vector.
        """
        inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
        
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        return text_features.cpu().detach().numpy().flatten()

# --- Quick Test Block ---
if __name__ == "__main__":
    # Initialize Engine
    engine = EmbeddingEngine()
    
    # Test with one frame from previous step
    test_image = "data/frames/frame_0.0.jpg" 
    
    if os.path.exists(test_image):
        vector = engine.get_image_embedding(test_image)
        print(f"\n🔮 Vector Generated!")
        print(f"Dimensions: {len(vector)}") # Should be 512
        print(f"First 5 values: {vector[:5]}")
        
        # Test Text Embedding too
        text_vector = engine.get_text_embedding("A photo of a video frame")
        print(f"📝 Text Vector Generated! First 5 values: {text_vector[:5]}")
    else:
        print(f"⚠️ Please verify '{test_image}' exists to run the test.")
