import streamlit as st
import tempfile
from processors import analyze_video
from deepstar_plugin.adapter import benchmark

st.set_page_config(page_title="Deepfake Analyzer")
st.title("Deepfake Detection Portal")

uploaded_file = st.file_uploader("Upload video", type=["mp4", "mov"])
if uploaded_file:
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())
    
    with st.spinner('Analyzing frames...'):
        result = analyze_video(tfile.name)
        deepstar_score = benchmark(tfile.name)
        
    st.subheader("Results:")
    col1, col2 = st.columns(2)
    col1.metric("Detection Confidence", f"{result['confidence']*100:.2f}%")
    col2.metric("Deepstar Compatibility", f"{deepstar_score*100:.2f}%")
    
    st.video(uploaded_file)
