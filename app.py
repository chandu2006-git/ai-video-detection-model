import os
import streamlit as st
import plotly.graph_objects as go

from services.predictor import predict_video

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="AI Video Detection Laboratory",
    page_icon="🎥",
    layout="wide"
)

# =====================================================
# LOAD CSS
# =====================================================

with open("assets/styles.css", "r", encoding="utf-8") as f:
    st.markdown(
        f"<style>{f.read()}</style>",
        unsafe_allow_html=True
    )

# =====================================================
# SESSION STATE
# =====================================================

if "result" not in st.session_state:
    st.session_state.result = None

# =====================================================
# HEADER
# =====================================================

st.markdown("""
<div class="hero-container">

    <div class="hero-badge">
        VIDEO AUTHENTICITY ANALYSIS
    </div>

    <div class="hero-title">
        AI Video Detection Laboratory
    </div>

    <div class="hero-description">
        Analyze uploaded videos using CNN-LSTM spatial-temporal learning
        and identify potential AI-generated content.
    </div>

</div>
""", unsafe_allow_html=True)

# =====================================================
# MAIN LAYOUT
# =====================================================

workflow_col, upload_col, result_col = st.columns(
    [0.8, 1.5, 1.8],
    gap="large"
)

# =====================================================
# WORKFLOW SIDEBAR
# =====================================================

with workflow_col:

    st.markdown("""
    <div class="sidebar-card">

        <div class="sidebar-title">
            Workflow
        </div>

        <div class="workflow-item">① Upload Video</div>
        <div class="workflow-item">② Frame Extraction</div>
        <div class="workflow-item">③ CNN Features</div>
        <div class="workflow-item">④ LSTM Analysis</div>
        <div class="workflow-item">⑤ Final Decision</div>

        <hr>

        <div class="sidebar-title">
            Status
        </div>

        <div class="status-card">
            🟢 Model Ready
        </div>

        <div class="model-info">
            CNN + LSTM
        </div>

        <div class="model-version">
            Version 2.0
        </div>

    </div>
    """, unsafe_allow_html=True)

# =====================================================
# UPLOAD PANEL
# =====================================================

with upload_col:

    st.markdown("""
    <div class="panel-title">
        Upload & Analysis
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "",
        type=["mp4", "avi", "mov", "mkv"],
        label_visibility="collapsed"
    )

    video_path = None

    if uploaded_file:

        os.makedirs(
            "uploads",
            exist_ok=True
        )

        video_path = os.path.join(
            "uploads",
            uploaded_file.name
        )

        with open(video_path, "wb") as f:
            f.write(
                uploaded_file.getbuffer()
            )

    # ==========================================
    # VIDEO HOLDER
    # ==========================================

    st.markdown("""
    <div class="video-box">
    """, unsafe_allow_html=True)

    if uploaded_file:

        st.video(uploaded_file)

    else:

        st.markdown("""
        <div class="empty-video">
            Upload a video to preview
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # ==========================================
    # FILE INFO
    # ==========================================

    if uploaded_file:

        st.markdown(f"""
        <div class="file-card">
            <strong>{uploaded_file.name}</strong>
            <br>
            {(uploaded_file.size/1024/1024):.2f} MB
        </div>
        """, unsafe_allow_html=True)

    # ==========================================
    # ANALYZE BUTTON
    # ==========================================

    analyze_btn = st.button(
        "Analyze Video",
        use_container_width=True
    )

    if analyze_btn and video_path:

        with st.spinner("Analyzing Video..."):

            st.session_state.result = predict_video(
                video_path
            )

# =====================================================
# RESULTS PANEL
# =====================================================

with result_col:

    st.markdown("""
    <div class="panel-title">
        Analysis Results
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.result is None:

        st.info(
            "Upload and analyze a video."
        )

    else:

        result = st.session_state.result

        label = result["label"]

        ai_prob = float(
            result["ai_probability"]
        )

        real_prob = round(
            100 - ai_prob,
            2
        )

        confidence = max(
            ai_prob,
            real_prob
        )

        # ==================================
        # VERDICT
        # ==================================

        if label == "REAL VIDEO":

            st.markdown(
                f"""
                <div class="result-real">
                REAL VIDEO
                </div>
                """,
                unsafe_allow_html=True
            )

        elif label == "AI GENERATED":

            st.markdown(
                f"""
                <div class="result-ai">
                AI GENERATED
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                f"""
                <div class="result-uncertain">
                UNCERTAIN
                </div>
                """,
                unsafe_allow_html=True
            )

        st.write("")

        # ==================================
        # DONUT CHART
        # ==================================

        fig = go.Figure(
            data=[
                go.Pie(
                    labels=["Real", "AI"],
                    values=[
                        real_prob,
                        ai_prob
                    ],
                    hole=0.65
                )
            ]
        )

        fig.update_traces(
            marker=dict(
                colors=[
                    "#10B981",
                    "#6D28D9"
                ]
            )
        )

        fig.update_layout(
            height=320,
            margin=dict(
                l=10,
                r=10,
                t=10,
                b=10
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # ==================================
        # METRICS
        # ==================================

        m1, m2, m3 = st.columns(3)

        with m1:

            st.markdown(f"""
            <div class="metric-card">

                <div class="metric-value">
                    {real_prob:.2f}%
                </div>

                <div class="metric-label">
                    Real
                </div>

            </div>
            """, unsafe_allow_html=True)

        with m2:

            st.markdown(f"""
            <div class="metric-card">

                <div class="metric-value">
                    {ai_prob:.2f}%
                </div>

                <div class="metric-label">
                    AI
                </div>

            </div>
            """, unsafe_allow_html=True)

        with m3:

            st.markdown(f"""
            <div class="metric-card">

                <div class="metric-value">
                    {confidence:.2f}%
                </div>

                <div class="metric-label">
                    Confidence
                </div>

            </div>
            """, unsafe_allow_html=True)

        st.write("")

        # ==================================
        # INTERPRETATION
        # ==================================

        st.markdown(f"""
        <div class="interpretation-card">

            <strong>Interpretation</strong>

            <br><br>

            Model confidence currently favors
            <b>{label}</b> based on extracted
            spatial-temporal characteristics.

        </div>
        """, unsafe_allow_html=True)

# =====================================================
# FOOTER
# =====================================================

st.markdown("""
<div class="footer">
    AI Video Detection Laboratory • CNN-LSTM Framework
</div>
""", unsafe_allow_html=True)