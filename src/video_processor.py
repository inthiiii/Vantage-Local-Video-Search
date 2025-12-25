import cv2
import os
from typing import List, Tuple
from PIL import Image

class VideoProcessor:
    """
    Handles the ingestion of video files and extraction of keyframes 
    for downstream AI processing.
    """
    
    def __init__(self, output_folder: str = "data/frames"):
        self.output_folder = output_folder
        # Create output directory if it doesn't exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

    def extract_keyframes(self, video_path: str, interval: int = 2) -> List[Tuple[str, float]]:
        """
        Extracts one frame every 'interval' seconds.
        
        Args:
            video_path (str): Path to the source video file.
            interval (int): Seconds between extracted frames.
            
        Returns:
            List[Tuple[str, float]]: A list of tuples containing (saved_image_path, timestamp).
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps * interval)
        
        extracted_frames = []
        frame_count = 0
        saved_count = 0

        print(f"🎬 Processing {video_path}...")
        
        while True:
            success, frame = cap.read()
            if not success:
                break
            
            # Only save the frame if it matches our interval
            if frame_count % frame_interval == 0:
                # Calculate timestamp
                timestamp = frame_count / fps
                
                # Construct filename: frame_0.0s.jpg, frame_2.0s.jpg, etc.
                filename = f"frame_{timestamp:.1f}.jpg"
                save_path = os.path.join(self.output_folder, filename)
                
                # Save frame
                cv2.imwrite(save_path, frame)
                extracted_frames.append((save_path, timestamp))
                saved_count += 1
                
            frame_count += 1

        cap.release()
        print(f"✅ Extracted {saved_count} keyframes to '{self.output_folder}'")
        return extracted_frames

# --- Quick Test Block (Runs only if you run this script directly) ---
if __name__ == "__main__":
    # 1. Put a short video (e.g., test.mp4) in your 'assets' folder
    # 2. Run this script: python src/video_processor.py
    
    # Create a dummy video path for testing logic (Replace with your real video path)
    test_video = "assets/test_video.mp4" 
    
    if os.path.exists(test_video):
        processor = VideoProcessor()
        frames = processor.extract_keyframes(test_video, interval=2)
        print(f"First 3 frames: {frames[:3]}")
    else:
        print("⚠️ Please place a video file in 'assets/' to test.")
