{\rtf1\ansi\ansicpg1252\cocoartf2867
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx566\tx1133\tx1700\tx2267\tx2834\tx3401\tx3968\tx4535\tx5102\tx5669\tx6236\tx6803\pardirnatural\partightenfactor0

\f0\fs24 \cf0 # \uc0\u55357 \u56385 \u65039  Vantage: Local Multimodal Video Search Engine\
\
**Vantage** is a privacy-first, offline AI tool that allows users to perform semantic search inside video files. By leveraging **OpenAI's CLIP** (Computer Vision) and **FAISS** (Vector Search), it enables users to find specific moments in a video using natural language queries (e.g., *"Show me the red car"* or *"Where is the graph shown?"*).\
\
Built to run entirely on the **Apple M3 Neural Engine** with zero cloud dependencies.\
\
## \uc0\u55357 \u56960  Key Features\
* **Multimodal RAG:** Maps video frames and text queries into a shared vector space (512-dim).\
* **Privacy-First:** 100% offline execution; no data leaves the machine.\
* **Hardware Accelerated:** Optimized with PyTorch MPS (Metal Performance Shaders) for M3 chips.\
* **Sub-Second Retrieval:** Uses FAISS for high-performance similarity search across thousands of frames.\
\
## \uc0\u55357 \u57056 \u65039  Tech Stack\
* **Core Logic:** Python 3.10\
* **AI Models:** OpenAI CLIP (`vit-base-patch32`)\
* **Vector Database:** FAISS (Facebook AI Similarity Search)\
* **Hardware Acceleration:** PyTorch MPS (Apple Silicon)\
* **UI:** Streamlit\
\
## \uc0\u9889  How to Run\
1.  Clone the repo:\
    ```bash\
    git clone [https://github.com/yourusername/Vantage-Local-Video-Search.git](https://github.com/yourusername/Vantage-Local-Video-Search.git)\
    cd Vantage\
    ```\
2.  Install dependencies:\
    ```bash\
    pip install -r requirements.txt\
    ```\
3.  Launch the App:\
    ```bash\
    streamlit run app.py\
    ```}