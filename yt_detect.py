import re
import streamlit as st
from google import genai
from google.genai import types
import json
import time
import os
import yt_dlp # The magic library for YouTube

# --- CONFIGURATION ---
# Streamlit automatically looks in secrets.toml (local) or Cloud Secrets (deployed)
try:
    API_KEY = st.secrets["GOOGLE_API_KEY"]
except FileNotFoundError:
    st.error("Secrets file not found. Please create .streamlit/secrets.toml")
    st.stop()

# We keep our dual-expert setup
VIDEO_MODEL_ID = "gemini-2.5-pro" 
AUDIO_MODEL_ID = "gemini-2.0-flash"

client = genai.Client(api_key=API_KEY)

# --- HELPER FUNCTIONS ---

def download_youtube_video(url):
    """
    Downloads video to 'downloads/' folder using the YouTube ID as the filename.
    Checks if it already exists to avoid redownloading.
    """
    # 1. Extract the Video ID using Regex (faster than spinning up yt-dlp)
    # This grabs the 'v' parameter from url (e.g., v=dQw4w9WgXcQ)
    video_id_match = re.search(r"(?:v=|\/)([\w-]{11})(?:\?|&|\/|$)", url)
    if not video_id_match:
        raise ValueError("Could not extract video ID from URL")
    
    video_id = video_id_match.group(1)
    
    # 2. Define the permanent path
    # We use the ID so 'temp_video.mp4' doesn't get overwritten
    output_filename = f"downloads/{video_id}.mp4"
    
    # 3. Create downloads folder if it doesn't exist
    if not os.path.exists("downloads"):
        os.makedirs("downloads")

    # 4. THE SMART CHECK: If file exists, return it immediately
    if os.path.exists(output_filename):
        st.toast(f"Found {video_id} in cache. Skipping download!", icon="⚡")
        return output_filename

    # 5. If not found, download it
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        # Save explicitly as "downloads/VIDEO_ID.mp4"
        'outtmpl': f'downloads/{video_id}.%(ext)s', 
        'quiet': True,
        'no_warnings': True,
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
        
    return output_filename

def upload_to_gemini(file_path, mime_type="video/mp4"):
    """Uploads local file path to Gemini."""
    print(f"Uploading {file_path}...")
    
    # FIXED: The argument is 'file', not 'path'
    video_file = client.files.upload(file=file_path)
    
    while video_file.state == "PROCESSING":
        time.sleep(1)
        video_file = client.files.get(name=video_file.name)
        
    if video_file.state == "FAILED":
        raise ValueError("Video processing failed at Google's end.")
        
    return video_file

def analyze_visuals(file_obj):
    schema = {
        "type": "OBJECT",
        "properties": {
            "visual_score": {"type": "INTEGER", "description": "0-100 authenticity score."},
            "visual_verdict": {"type": "STRING", "enum": ["Real", "Fake", "Uncertain"]},
            "visual_anomalies": {
                "type": "ARRAY", 
                "items": {"type": "OBJECT", "properties": {"time": {"type": "STRING"}, "desc": {"type": "STRING"}}}
            }
        }
    }
    prompt = "Analyze frame-by-frame for deepfake artifacts, unnatural blinking, and shadow errors."
    
    response = client.models.generate_content(
        model=VIDEO_MODEL_ID,
        contents=[
            types.Content(
                role="user",
                parts=[
                    types.Part.from_uri(file_uri=file_obj.uri, mime_type=file_obj.mime_type),
                    types.Part.from_text(text=prompt)
                ]
            )
        ],
        config=types.GenerateContentConfig(response_mime_type="application/json", response_schema=schema)
    )
    return json.loads(response.text)

def analyze_audio(file_obj):
    schema = {
        "type": "OBJECT",
        "properties": {
            "audio_score": {"type": "INTEGER", "description": "0-100 authenticity score."},
            "audio_verdict": {"type": "STRING", "enum": ["Natural", "Synthetic", "Mixed/Edited"]},
            "acoustic_analysis": {"type": "STRING"},
            "detected_issues": {"type": "ARRAY", "items": {"type": "STRING"}}
        }
    }
    prompt = "Listen for AI voice artifacts: metallic robotic ends of words, flat pitch, lack of breath."

    response = client.models.generate_content(
        model=AUDIO_MODEL_ID,
        contents=[
            types.Content(
                role="user",
                parts=[
                    types.Part.from_uri(file_uri=file_obj.uri, mime_type=file_obj.mime_type),
                    types.Part.from_text(text=prompt)
                ]
            )
        ],
        config=types.GenerateContentConfig(response_mime_type="application/json", response_schema=schema)
    )
    return json.loads(response.text)

# --- UI ---

st.set_page_config(page_title="Deepfake Inspector", layout="wide")
st.title("🕵️‍♂️ Semantic Deepfake Inspector")

# Choose Input Method
input_method = st.radio("Source:", ["📺 YouTube URL", "📁 File Upload"], horizontal=True)

target_file_path = None
process_trigger = False

# INPUT LOGIC
if input_method == "📺 YouTube URL":
    yt_url = st.text_input("Paste YouTube Link", placeholder="https://www.youtube.com/watch?v=...")
    if yt_url and st.button("🚀 Analyze YouTube Video"):
        with st.spinner("Downloading video from YouTube (Background)..."):
            try:
                target_file_path = download_youtube_video(yt_url)
                process_trigger = True
            except Exception as e:
                st.error(f"Could not download video: {e}")

elif input_method == "📁 File Upload":
    uploaded_file = st.file_uploader("Upload Video", type=["mp4", "mov"])
    if uploaded_file and st.button("🚀 Analyze File"):
        # Save uploaded file to disk
        target_file_path = "temp_upload.mp4"
        with open(target_file_path, "wb") as f:
            f.write(uploaded_file.read())
        process_trigger = True

# ANALYSIS LOGIC
if process_trigger and target_file_path:
    # FIXED: Use columns to restrict the video width
    # This creates 3 columns: Empty (1) | Video (2) | Empty (1) -> Centers and shrinks video
    col1, col_video, col2 = st.columns([1, 2, 1])
    
    with col_video:
        st.video(target_file_path)
    
    try:
        with st.spinner("Uploading to Gemini & Running Dual-Forensics..."):
            # Upload
            gemini_file = upload_to_gemini(target_file_path)
            
            # Analyze (Parallel calls would be faster, but sequential is safer for rate limits)
            visual_res = analyze_visuals(gemini_file)
            audio_res = analyze_audio(gemini_file)
            
            # --- RESULTS DASHBOARD ---
            st.divider()
            col1, col2 = st.columns(2)
            
            # Visual Report
            with col1:
                v_score = visual_res.get("visual_score", 0)
                v_color = "green" if v_score > 80 else "red"
                st.markdown(f"### 👁️ Visual: <span style='color:{v_color}'>{v_score}/100</span>", unsafe_allow_html=True)
                st.caption(visual_res.get("visual_verdict"))
                
                anomalies = visual_res.get("visual_anomalies", [])
                if anomalies:
                    for a in anomalies:
                        st.warning(f"**{a['time']}**: {a['desc']}")
                else:
                    st.success("No visual anomalies detected.")

            # Audio Report
            with col2:
                a_score = audio_res.get("audio_score", 0)
                a_color = "green" if a_score > 80 else "red"
                st.markdown(f"### 👂 Audio: <span style='color:{a_color}'>{a_score}/100</span>", unsafe_allow_html=True)
                st.caption(audio_res.get("audio_verdict"))
                
                st.info(f"**Analysis:** {audio_res.get('acoustic_analysis')}")
                for issue in audio_res.get("detected_issues", []):
                    st.error(f"Detected: {issue}")

    except Exception as e:
        st.error(f"Analysis failed: {e}")
        
    #finally:
    #    Cleanup temp files
    #    if os.path.exists(target_file_path):
    #        os.remove(target_file_path)