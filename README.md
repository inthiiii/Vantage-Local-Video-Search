# 👁️ Vantage: Local Multimodal Video Search Engine

**Vantage** is a privacy-first, offline AI tool that allows users to perform semantic search inside video files. By leveraging **OpenAI's CLIP** (Computer Vision) and **FAISS** (Vector Search), it enables users to find specific moments in a video using natural language queries (e.g., *"Show me the red car"* or *"Where is the graph shown?"*).

Built to run entirely on the **Apple M3 Neural Engine** with zero cloud dependencies.

## 🚀 Key Features
* **Multimodal RAG:** Maps video frames and text queries into a shared vector space (512-dim).
* **Privacy-First:** 100% offline execution; no data leaves the machine.
* **Hardware Accelerated:** Optimized with PyTorch MPS (Metal Performance Shaders) for M3 chips.
* **Sub-Second Retrieval:** Uses FAISS for high-performance similarity search across thousands of frames.

## 🛠️ Tech Stack
* **Core Logic:** Python 3.10
* **AI Models:** OpenAI CLIP (`vit-base-patch32`)
* **Vector Database:** FAISS (Facebook AI Similarity Search)
* **Hardware Acceleration:** PyTorch MPS (Apple Silicon)
* **UI:** Streamlit

## ⚡ How to Run

1. Clone the repo:
   ```bash
   git clone [https://github.com/yourusername/Vantage-Local-Video-Search.git](https://github.com/yourusername/Vantage-Local-Video-Search.git)
   cd Vantage