import streamlit as st
import os
from src.video_processor import VideoProcessor
from src.vector_db import VectorDB
from PIL import Image

# Page Config
st.set_page_config(page_title="Vantage | Video Search", layout="wide")

st.title("👁️ Vantage")
st.caption("The Local Multimodal Video Search Engine (Powered by M3, CLIP & FAISS)")

# Initialize Backend
if 'processor' not in st.session_state:
    st.session_state.processor = VideoProcessor()
if 'db' not in st.session_state:
    st.session_state.db = VectorDB()

# Sidebar: Upload
with st.sidebar:
    st.header("1. Ingest Video")
    uploaded_file = st.file_uploader("Upload a video (MP4)", type=['mp4'])
    
    if uploaded_file:
        # Save temp file
        temp_path = os.path.join("assets", "temp_video.mp4")
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success("Video uploaded!")
        
        if st.button("Analyze Video"):
            with st.spinner("Processing Frames & Vectors (M3 Neural Engine)..."):
                # 1. Extract Frames
                frames = st.session_state.processor.extract_keyframes(temp_path)
                
                # 2. Build Index
                st.session_state.db.build_index("data/frames")
                
            st.success("✅ Analysis Complete! You can now search.")

# Main Area: Search
st.header("2. Search Inside Video")
query = st.text_input("What are you looking for?", placeholder="e.g., 'red car', 'people laughing', 'a graph'")

if query:
    results = st.session_state.db.search(query, k=3)
    
    if results:
        st.subheader(f"Found {len(results)} matches for '{query}':")
        
        cols = st.columns(3)
        for i, match_path in enumerate(results):
            # Parse timestamp from filename (frame_2.0.jpg -> 2.0s)
            filename = os.path.basename(match_path)
            timestamp = filename.split('_')[1].replace('.jpg', '')
            
            with cols[i]:
                img = Image.open(match_path)
                st.image(img, caption=f"Timestamp: {timestamp}s")
                st.info(f"Jump to {timestamp}s")
    else:
        st.warning("No matches found.")
