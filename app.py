import streamlit as st
import os
from src.video_processor import VideoProcessor
from src.vector_db import VectorDB
from PIL import Image

# --- 1. CONFIGURATION ---
st.set_page_config(
    page_title="Vantage | AI Video Search",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 2. PREMIUM CSS & UI FIXES ---
st.markdown("""
<style>
    /* 1. Global Theme - Dark Mode Base */
    .stApp {
        background-color: #0E1117;
    }
    
    /* 2. Text Hierarchy (Not everything should be pure white) */
    h1, h2, h3 {
        color: #FFFFFF !important;
        font-family: 'SF Pro Display', sans-serif;
        letter-spacing: -0.5px;
    }
    p, span, div {
        color: #E0E0E0; /* Soft White for better reading */
        font-family: 'SF Pro Text', sans-serif;
    }
    small, .caption {
        color: #A0A0A0 !important; /* Dim Grey for secondary info */
    }
    
    /* 3. Navigation/Sidebar Fixes */
    section[data-testid="stSidebar"] {
        background-color: #161B22; /* Slightly lighter than main bg */
        border-right: 1px solid #30363D;
    }
    
    /* 4. File Uploader Visibility Fix (Crucial) */
    [data-testid="stFileUploader"] {
        background-color: #21262D; /* Dark Container */
        border: 1px dashed #30363D;
        border-radius: 10px;
        padding: 20px;
    }
    [data-testid="stFileUploader"] div {
        color: #FFFFFF !important; /* Force drag-drop text to be white */
    }
    [data-testid="stFileUploader"] small {
        color: #8B949E !important; /* Force file limits text to be grey */
    }
    
    /* 5. Input Fields */
    .stTextInput input {
        background-color: #0D1117;
        color: #FFFFFF;
        border: 1px solid #30363D;
        border-radius: 8px;
    }
    
    /* 6. Buttons - Primary Action */
    .stButton>button {
        background: linear-gradient(180deg, #238636 0%, #2EA043 100%); /* GitHub Green style */
        color: white !important;
        border: 1px solid rgba(240,246,252,0.1);
        border-radius: 6px;
        font-weight: 600;
        box-shadow: 0 1px 0 rgba(27,31,35,0.1);
    }
    
    /* 7. Result Cards */
    .result-timestamp {
        background-color: #1F6FEB; /* Blue Badge */
        color: white;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.85em;
        font-weight: bold;
        display: inline-block;
        margin-bottom: 8px;
    }
    
    /* Hide Default Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
</style>
""", unsafe_allow_html=True)

# --- 3. BACKEND INIT ---
if 'processor' not in st.session_state:
    st.session_state.processor = VideoProcessor()
if 'db' not in st.session_state:
    st.session_state.db = VectorDB()

if not os.path.exists("assets"):
    os.makedirs("assets")

# --- 4. SIDEBAR (CONTEXT) ---
with st.sidebar:
    st.title("👁️ Vantage")
    st.caption("v1.2")
    st.markdown("---")
    
    st.markdown("### 💡 Pro Tips")
    st.info("""
    **Be Specific:**
    Instead of *"car"*, try:
    - *"Vintage red car on a dirt road"*
    - *"Bar chart showing growth"*
    - *"Close up of a person smiling"*
    """)
    
    st.markdown("### ⚙️ Engine Status")
    st.success("M3 Neural Engine: **Ready**")

# --- 5. MAIN LAYOUT ---

# Top Header
st.title("Vantage  Vision")
st.markdown("<h3 style='color: #8B949E !important; margin-top: -20px;'>Local Multimodal Video Search Engine</h3>", unsafe_allow_html=True)
st.markdown("---")

# DASHBOARD CONTROLS (Compact Row)
c1, c2 = st.columns([2, 1], gap="large")
video_path = os.path.join("assets", "temp_video.mp4")

with c1:
    # The uploader now has a dark background via CSS
    uploaded_file = st.file_uploader("1. Drop Video Here (MP4)", type=['mp4'])
    if uploaded_file:
        with open(video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

with c2:
    st.markdown("<div style='height: 29px'></div>", unsafe_allow_html=True) # Spacer
    if uploaded_file:
        if st.button("2. Initialize AI Engine ⚡", type="primary", use_container_width=True):
            with st.status("Processing Content...", expanded=True) as status:
                st.write("🎞️ Extracting frames (OpenCV)...")
                st.session_state.processor.extract_keyframes(video_path)
                st.write("🧠 Vectorizing visuals (CLIP ViT-B/32)...")
                st.session_state.db.build_index("data/frames")
                status.update(label="✅ Indexing Complete!", state="complete", expanded=False)
            st.rerun()
    else:
        st.button("Waiting for Video...", disabled=True, use_container_width=True)

# --- 6. RESULTS INTERFACE ---
is_ready = os.path.exists(video_path) and os.path.exists("data/vector_index.bin")

if is_ready:
    st.markdown("---")
    
    # Split Layout
    col_search, col_player = st.columns([1.5, 1], gap="large")
    
    # LEFT: SEARCH
    with col_search:
        st.subheader("🔍 Semantic Search")
        query = st.text_input("Search", placeholder="Describe a moment (e.g., 'explosion', 'handshake')...", label_visibility="collapsed")
        
        if query:
            results = st.session_state.db.search(query, k=3)
            
            if results:
                st.markdown("### Top Matches")
                # Grid for results
                r_cols = st.columns(3)
                
                for i, match_path in enumerate(results):
                    filename = os.path.basename(match_path)
                    try:
                        timestamp = float(filename.split('_')[1].replace('.jpg', ''))
                    except:
                        timestamp = 0.0
                    
                    with r_cols[i]:
                        img = Image.open(match_path)
                        st.image(img, use_container_width=True)
                        
                        # Custom Timestamp Badge
                        st.markdown(f"<div class='result-timestamp'>⏱ {timestamp}s</div>", unsafe_allow_html=True)
                        
                        if st.button(f"▶ Play", key=f"play_{i}", use_container_width=True):
                            st.session_state['current_video_time'] = timestamp
                            st.rerun()
            else:
                st.warning("No matches found. Try a different query.")
        else:
             st.markdown("""
             <div style='background-color: #161B22; padding: 20px; border-radius: 10px; border: 1px solid #30363D;'>
                <p style='margin:0; color: #8B949E;'>Enter a description above to search inside the video.</p>
             </div>
             """, unsafe_allow_html=True)

    # RIGHT: PLAYER
    with col_player:
        st.subheader("🎬 Media Player")
        start_time = st.session_state.get('current_video_time', 0)
        st.video(video_path, start_time=int(start_time))
        st.caption(f"Playback synchronized at: {start_time}s")

else:
    # EMPTY STATE (Clean)
    st.markdown("""
    <br>
    <div style='text-align: center; color: #8B949E;'>
        <h3>👈 Upload a video to begin</h3>
        <p>Your data stays 100% offline.</p>
    </div>
    """, unsafe_allow_html=True)