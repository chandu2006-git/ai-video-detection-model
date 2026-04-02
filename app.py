import streamlit as st
import tempfile
from services.predictor import predict_video

st.set_page_config(
    page_title="AI Video Detection",
    layout="centered"
)

st.title("🎥 AI Video Detection System")
st.write("Upload a video to check whether it is **AI-generated** or **Real**.")

uploaded_video = st.file_uploader(
    "Upload Video",
    type=["mp4", "avi", "mov"]
)

if uploaded_video is not None:
    st.video(uploaded_video)

    if st.button("🔍 Predict"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded_video.read())
            video_path = tmp.name

        with st.spinner("Analyzing video..."):
            result = predict_video(video_path)

        st.success("✅ Prediction Completed")

        st.subheader("Result")
        st.markdown(f"### **{result['label']}**")
        st.markdown(f"**AI Probability:** `{result['ai_probability']} %`")
