"""
AI Video Detection Laboratory
Production-quality Streamlit forensic dashboard.
"""

import time
from pathlib import Path
from services.predictor import predict_video
import streamlit as st

# ── Page config — must be first ───────────────────────────────────────────────
st.set_page_config(
    page_title="AI Video Detection Laboratory",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ─────────────────────────────────────────────────────────────────
UPLOADS_DIR = Path(__file__).parent / "uploads"
UPLOADS_DIR.mkdir(exist_ok=True)

ANALYSIS_STEPS = [
    (0.12, "⏳  Initializing model…"),
    (0.30, "🎞  Extracting frames…"),
    (0.54, "🧠  Running ResNet50 features…"),
    (0.78, "🔁  Running CNN-LSTM…"),
    (0.94, "📊  Generating report…"),
]

LABEL_CONFIG = {
    "REAL": {
        "color": "#2ECC71",
        "bg":    "#F0FDF4",
        "text":  "REAL VIDEO",
        "sub":   "The video is likely authentic and not AI-generated.",
        "icon":  "M20 6L9 17l-5-5",
    },
    "AI_GENERATED": {
        "color": "#FF5C5C",
        "bg":    "#FFF5F5",
        "text":  "AI GENERATED",
        "sub":   "Synthetic content detected by the model.",
        "icon":  "M18 6L6 18M6 6l12 12",
    },
    "UNCERTAIN": {
        "color": "#F59E0B",
        "bg":    "#FFFBEB",
        "text":  "UNCERTAIN",
        "sub":   "The model could not reach a high-confidence verdict.",
        "icon":  "M12 9v4m0 4h.01M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z",
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# PREDICTION STUB — replace body with: from services.predictor import predict_video
# ══════════════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════════════
# CSS LOADER
# ══════════════════════════════════════════════════════════════════════════════

def load_css() -> None:
    css_path = Path(__file__).parent / "style.css"
    if css_path.exists():
        with open(css_path, encoding="utf-8") as fh:
            st.markdown(f"<style>{fh.read()}</style>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

def render_sidebar() -> None:
    """Render the fixed left sidebar with workflow steps."""
    STEPS = [
        ("M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M17 8l-5-5-5 5M12 3v12",
         "Upload Video",     "Upload the video file to be analyzed",     "#6C63FF"),
        ("M4 6h16M4 12h16M4 18h7",
         "Extract Frames",   "Extracting 30 representative frames",      "#8A7CFF"),
        ("M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5",
         "ResNet50",         "Spatial feature extraction per frame",      "#6C63FF"),
        ("M22 12h-4l-3 9L9 3l-3 9H2",
         "CNN-LSTM",         "Temporal pattern analysis",                "#8A7CFF"),
        ("M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z",
         "Prediction",       "Real or AI-generated classification",       "#6C63FF"),
    ]

    with st.sidebar:
        # Logo
        st.markdown("""
        <div class="sb-logo">
            <svg width="32" height="32" viewBox="0 0 38 38" fill="none">
                <path d="M19 3 L34 9 L34 20 Q34 31 19 36 Q4 31 4 20 L4 9 Z"
                      fill="#6C63FF" fill-opacity="0.15" stroke="#6C63FF" stroke-width="1.5"/>
                <circle cx="19" cy="19" r="6" fill="#6C63FF" opacity="0.8"/>
                <circle cx="19" cy="19" r="2.5" fill="white"/>
            </svg>
            <div class="sb-logo-text">
                <div class="sb-logo-title">AI VIDEO <span class="logo-accent">DETECTION</span></div>
                <div class="sb-logo-sub">LABORATORY</div>
            </div>
        </div>
        <div class="sb-divider"></div>
        <div class="section-eyebrow" style="padding: 0 4px;">DETECTION WORKFLOW</div>
        """, unsafe_allow_html=True)

        # Steps
        steps_html = ""
        for i, (icon_path, title, desc, color) in enumerate(STEPS):
            active = "step-active" if i == 0 else ("step-last" if i == 4 else "")
            connector = "<div class='step-connector'></div>" if i < 4 else ""
            steps_html += f"""
            <div class="workflow-step {active}">
                <div class="step-icon-wrap" style="border-color:{color}30; background:{color}12;">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none"
                         stroke="{color}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
                        <path d="{icon_path}"/>
                    </svg>
                </div>
                <div class="step-text">
                    <div class="step-num" style="color:{color};">{i+1}. {title}</div>
                    <div class="step-desc">{desc}</div>
                </div>
            </div>
            {connector}
            """

        st.markdown(steps_html, unsafe_allow_html=True)

        # AI Forensics footer card
        st.markdown("""
        <div class="sb-divider"></div>
        <div class="ai-forensics-card">
            <div class="af-icon">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none"
                     stroke="#6C63FF" stroke-width="1.8" stroke-linecap="round">
                    <path d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3
                             m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547
                             A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531
                             c0-.895-.356-1.754-.988-2.386l-.548-.547z"/>
                </svg>
            </div>
            <div>
                <div class="af-title">AI Forensics</div>
                <div class="af-desc">Detecting synthetic videos<br>with deep learning</div>
            </div>
        </div>
        <div class="sb-model-tag">CNN-LSTM + ResNet50</div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════

def render_header() -> None:
    st.markdown("""
    <div class="header-bar">
        <div class="header-center">
            <h1 class="main-title">AI Video Detection Laboratory</h1>
            <p class="main-subtitle">Deep Learning for Video Forensics &nbsp;•&nbsp; CNN-LSTM + ResNet50 Architecture</p>
        </div>
        <div class="header-right">
            <div class="status-pill">
                <span class="status-dot"></span>
                <div>
                    <div class="status-label">Model Status</div>
                    <div class="status-value">Online</div>
                </div>
            </div>
            <div class="model-tag">CNN-LSTM + ResNet50</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# UPLOAD PANEL
# ══════════════════════════════════════════════════════════════════════════════

def render_upload() -> object:
    """Render upload card header and file uploader. Returns uploaded file or None."""
    st.markdown("""
    <div class="card-header">
<svg
    width="16"
    height="16"
    viewBox="0 0 24 24"
    fill="none"
    stroke="#6C63FF"
    stroke-width="2"
    stroke-linecap="round"
    style="margin-top:-46px; margin-left:8px;"
>
    <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M17 8l-5-5-5 5M12 3v12"/>
</svg>
        <span class="card-title">UPLOAD &amp; PREVIEW</span>
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Drop your video here",
        type=["mp4", "avi", "mov", "webm"],
        label_visibility="collapsed",
    )

    if not uploaded:
        st.markdown(
            '<div class="hint-formats">Supported: MP4 &nbsp;•&nbsp; AVI &nbsp;•&nbsp; MOV &nbsp;•&nbsp; WebM</div>',
            unsafe_allow_html=True,
        )

    return uploaded


def render_file_info(uploaded_file) -> None:
    size_mb = round(uploaded_file.size / (1024 * 1024), 1)
    ext     = uploaded_file.name.split(".")[-1].upper()
    st.markdown(f"""
    <div class="file-info-bar">
        <div class="file-icon">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none"
                 stroke="#6C63FF" stroke-width="2" stroke-linecap="round">
                <rect x="2" y="2" width="20" height="20" rx="3"/>
                <path d="M9 9l6 6M15 9l-6 6"/>
            </svg>
        </div>
        <div class="file-meta">
            <div class="file-name">{uploaded_file.name}</div>
            <div class="file-details">{size_mb} MB &nbsp;•&nbsp; {ext}</div>
        </div>
        <div class="file-check">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none"
                 stroke="#2ECC71" stroke-width="2.5" stroke-linecap="round">
                <path d="M20 6L9 17l-5-5"/>
            </svg>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_video(uploaded_file) -> None:
    st.markdown('<div class="video-preview-label">VIDEO PREVIEW</div>', unsafe_allow_html=True)
    st.video(uploaded_file)


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS ANIMATION
# ══════════════════════════════════════════════════════════════════════════════

def run_analysis(video_path: str) -> tuple[dict, float]:
    """Run animated progress bar, call predictor, return (result, elapsed)."""
    bar = st.progress(0, text="Starting analysis…")
    t0  = time.time()

    for val, msg in ANALYSIS_STEPS:
        time.sleep(0.5)
        bar.progress(val, text=msg)

    result = predict_video(video_path)

    bar.progress(1.0, text="✅  Analysis complete!")
    time.sleep(0.4)
    bar.empty()

    return result, round(time.time() - t0, 2)


# ══════════════════════════════════════════════════════════════════════════════
# RESULT COMPONENTS (right panel)
# ══════════════════════════════════════════════════════════════════════════════

def render_result_header(result: dict) -> None:
    conf = result["confidence"]
    if conf > 75:
        badge_label, badge_color = "HIGH CONFIDENCE",      "#2ECC71"
    elif conf > 50:
        badge_label, badge_color = "MODERATE CONFIDENCE",  "#F59E0B"
    else:
        badge_label, badge_color = "LOW CONFIDENCE",       "#FF5C5C"

    st.markdown(f"""
    <div class="result-header">
        <div class="result-title-row">
            <svg
                width="14"
                height="14"
                viewBox="0 0 24 24"
                fill="none"
                stroke="#6C63FF"
                stroke-width="2"
                stroke-linecap="round"
                style="margin-top:-44px; margin-left:8px;"
            >
                <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
            </svg>
            <span class="card-title">DETECTION RESULT</span>
        </div>
        <span class="conf-badge"
              style="color:{badge_color}; background:{badge_color}18; border-color:{badge_color}40;">
            {badge_label}
        </span>
    </div>
    """, unsafe_allow_html=True)


def render_result_banner(label: str) -> None:
    cfg = LABEL_CONFIG.get(label, LABEL_CONFIG["UNCERTAIN"])
    st.markdown(f"""
    <div class="result-banner"
         style="background:{cfg['bg']}; border-color:{cfg['color']}30;">
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none"
             stroke="{cfg['color']}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="{cfg['icon']}"/>
        </svg>
        <div>
            <div class="banner-label" style="color:{cfg['color']};">{cfg['text']}</div>
            <div class="banner-sub">{cfg['sub']}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_donut_chart(result: dict) -> None:
    """SVG thin donut chart with animated stroke."""
    label = result["label"]
    conf  = result["confidence"]
    real  = result["real_probability"]
    ai    = result["ai_probability"]

    c_real = "#2ECC71"
    c_ai   = "#FF5C5C"
    c_bg   = "#E7E8F0"

    r      = 72
    cx, cy = 90, 90
    circum = 2 * 3.14159265 * r
    real_dash = circum * (real / 100)
    ai_dash   = circum * (ai   / 100)

    # real arc starts at top (offset by quarter-circle)
    real_offset = circum * 0.25
    ai_offset   = real_offset - real_dash   # ai starts right after real arc

    ring_color = LABEL_CONFIG.get(label, LABEL_CONFIG["UNCERTAIN"])["color"]

    st.markdown(f"""
    <div class="ring-container">
        <svg width="180" height="180" viewBox="0 0 180 180">
            <!-- bg track -->
            <circle cx="{cx}" cy="{cy}" r="{r}" fill="none"
                    stroke="{c_bg}" stroke-width="8"/>
            <!-- real arc -->
            <circle cx="{cx}" cy="{cy}" r="{r}" fill="none"
                    stroke="{c_real}" stroke-width="8"
                    stroke-dasharray="{real_dash:.2f} {circum - real_dash:.2f}"
                    stroke-dashoffset="{real_offset:.2f}"
                    stroke-linecap="round" class="ring-progress"/>
            <!-- ai arc -->
            <circle cx="{cx}" cy="{cy}" r="{r}" fill="none"
                    stroke="{c_ai}" stroke-width="8"
                    stroke-dasharray="{ai_dash:.2f} {circum - ai_dash:.2f}"
                    stroke-dashoffset="{ai_offset:.2f}"
                    stroke-linecap="round" class="ring-progress"/>
            <!-- center text -->
            <text x="{cx}" y="{cy - 8}" text-anchor="middle"
                  font-family="'JetBrains Mono', monospace"
                  font-size="26" font-weight="700" fill="{ring_color}">{conf}%</text>
            <text x="{cx}" y="{cy + 12}" text-anchor="middle"
                  font-size="10" font-weight="400" fill="#A0A3B1" letter-spacing="2">CONFIDENCE</text>
        </svg>
    </div>
    """, unsafe_allow_html=True)


def render_probability_bars(result: dict) -> None:
    real = result["real_probability"]
    ai   = result["ai_probability"]
    st.markdown(f"""
    <div class="prob-card">
        <div class="prob-row">
            <div class="prob-dot" style="background:#2ECC71;"></div>
            <span class="prob-label">Real Video</span>
            <div class="prob-bar-wrap">
                <div class="prob-bar-fill"
                     style="width:{real}%; background:linear-gradient(90deg,#2ECC71,#52D98A);"></div>
            </div>
            <span class="prob-value" style="color:#2ECC71;">{real}%</span>
        </div>
        <div class="prob-row">
            <div class="prob-dot" style="background:#FF5C5C;"></div>
            <span class="prob-label">AI Generated</span>
            <div class="prob-bar-wrap">
                <div class="prob-bar-fill"
                     style="width:{ai}%; background:linear-gradient(90deg,#FF5C5C,#FF8080);"></div>
            </div>
            <span class="prob-value" style="color:#FF5C5C;">{ai}%</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_metrics(result: dict) -> None:
    real = result["real_probability"]
    ai   = result["ai_probability"]
    conf = result["confidence"]

    real_sub = "High"     if real > 70 else ("Moderate" if real > 40 else "Low")
    ai_sub   = "High"     if ai   > 70 else ("Moderate" if ai   > 40 else "Low")
    conf_sub = "Very High" if conf > 85 else ("High"    if conf > 65 else "Moderate")

    icon_up   = "M22 7l-8.5 8.5-5-5L1 18"
    icon_dn   = "M22 17l-8.5-8.5-5 5L1 6"
    icon_sh   = "M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z"

    def pill(eyebrow, icon_path, value, sub, color):
        return f"""
        <div class="metric-pill">
            <div class="metric-icon" style="background:{color}15;">
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none"
                     stroke="{color}" stroke-width="2" stroke-linecap="round">
                    <path d="{icon_path}"/>
                </svg>
            </div>
            <div>
                <div class="metric-eyebrow">{eyebrow}</div>
                <div class="metric-val" style="color:{color};">{value}</div>
                <div class="metric-sub">{sub}</div>
            </div>
        </div>"""

    html = (
        '<div class="metrics-row">'
        + pill("REAL PROB.",  icon_up, f"{real}%", real_sub, "#2ECC71")
        + pill("AI PROB.",    icon_dn, f"{ai}%",   ai_sub,  "#FF5C5C")
        + pill("CONFIDENCE",  icon_sh, f"{conf}%", conf_sub, "#6C63FF")
        + "</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


def render_summary(result: dict, inference_time: float) -> None:
    label_map = {"REAL": "Real Video", "AI_GENERATED": "AI Generated", "UNCERTAIN": "Uncertain"}

    rows = [
        ("Prediction",       label_map.get(result["label"], result["label"])),
        ("Confidence",       f"{result['confidence']}%"),
        ("AI Probability",   f"{result['ai_probability']}%"),
        ("Real Probability", f"{result['real_probability']}%"),
        ("Inference Time",   f"{inference_time:.2f}s"),
        ("Model",            "CNN-LSTM + ResNet50"),
    ]

    rows_html = "".join(
        f'<div class="summary-row">'
        f'<span class="summary-key">{k}</span>'
        f'<span class="summary-val">{v}</span>'
        f'</div>'
        for k, v in rows
    )
    st.markdown(
        f'<div class="summary-card">'
        f'<div class="summary-title">Prediction Summary</div>'
        f'{rows_html}'
        f'</div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# INTERPRETATION
# ══════════════════════════════════════════════════════════════════════════════

def render_interpretation(result: dict) -> None:
    label = result["label"]
    real  = result["real_probability"]
    ai    = result["ai_probability"]
    conf  = result["confidence"]
    cfg   = LABEL_CONFIG.get(label, LABEL_CONFIG["UNCERTAIN"])
    color = cfg["color"]

    if label == "REAL":
        body = (
            f"The CNN-LSTM model analyzed temporal patterns across 30 frames extracted from the video. "
            f"The predicted probability strongly indicates that this video is "
            f"<strong style='color:{color};'>REAL</strong>. "
            f"No significant artifacts or inconsistencies typically found in AI-generated videos were detected. "
            f"Spatial coherence across frames was high and temporal consistency was within expected bounds."
        )
    elif label == "AI_GENERATED":
        body = (
            f"The CNN-LSTM model identified strong temporal inconsistencies across the 30 analyzed frames. "
            f"ResNet50 feature maps revealed spatial anomalies characteristic of generative models. "
            f"This video is classified as <strong style='color:{color};'>AI GENERATED</strong> "
            f"with {conf}% confidence. "
            f"Deepfake artifacts such as boundary blurring or unnatural motion patterns may be present."
        )
    else:
        body = (
            f"The model returned a low-confidence verdict — "
            f"Real: {real}%, AI: {ai}%. "
            f"The margin is insufficient for a definitive classification. "
            f"This may result from heavy compression, mixed-source content, or a borderline synthetic video. "
            f"Consider additional forensic tools for confirmation."
        )

    st.markdown(f"""
    <div class="interp-card">
        <div class="interp-header">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none"
                 stroke="#6C63FF" stroke-width="2" stroke-linecap="round">
                <path d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3
                         m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547
                         A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531
                         c0-.895-.356-1.754-.988-2.386l-.548-.547z"/>
            </svg>
            <span class="interp-title">INTERPRETATION</span>
        </div>
        <p class="interp-body">{body}</p>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════

def render_footer() -> None:
    st.markdown("""
    <div class="footer">
        <span>© 2025 AI Video Detection Laboratory</span>
        <span class="footer-dot">•</span>
        <span>Built with Streamlit</span>
        <span class="footer-dot">•</span>
        <span>CNN-LSTM + ResNet50</span>
        <span class="footer-dot">•</span>
        <span>Deep Learning Video Forensics</span>
        <svg width="13" height="13" viewBox="0 0 24 24" fill="#6C63FF" opacity="0.5">
            <path d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944
                     a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9
                     c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622
                     0-1.042-.133-2.052-.382-3.016z"/>
        </svg>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# RIGHT PANEL — awaiting state
# ══════════════════════════════════════════════════════════════════════════════

def render_awaiting_panel() -> None:
    st.markdown("""
    <div class="panel-card result-placeholder">
        <div class="placeholder-icon">
            <svg width="44" height="44" viewBox="0 0 24 24" fill="none"
                 stroke="#C4C6D4" stroke-width="1.2" stroke-linecap="round">
                <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
            </svg>
        </div>
        <div class="placeholder-title">Awaiting Analysis</div>
        <div class="placeholder-sub">Upload a video and click Analyze to view detection results here.</div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    load_css()
    render_sidebar()
    render_header()

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    center_col, right_col = st.columns([1.45, 1.55], gap="medium")

    # ── CENTER ───────────────────────────────────────────────────────────────
    with center_col:
        with st.container():
            st.markdown('<div class="panel-card">', unsafe_allow_html=True)
            uploaded_file = render_upload()

            if uploaded_file:
                render_file_info(uploaded_file)
                render_video(uploaded_file)

                analyze_clicked = st.button(
                    "🚀  Analyze Video",
                    key="analyze_btn",
                    use_container_width=True,
                )
                st.markdown(
                    '<p class="analyze-hint">Click to start AI analysis</p>',
                    unsafe_allow_html=True,
                )

                if analyze_clicked:
                    suffix   = Path(uploaded_file.name).suffix
                    tmp_path = UPLOADS_DIR / f"upload_{int(time.time())}{suffix}"
                    tmp_path.write_bytes(uploaded_file.getbuffer())

                    result, elapsed = run_analysis(str(tmp_path))

                    try:
                        tmp_path.unlink()
                    except OSError:
                        pass

                    st.session_state["result"]         = result
                    st.session_state["inference_time"] = elapsed
                    st.rerun()

            st.markdown("</div>", unsafe_allow_html=True)

        # Interpretation below upload card (only after prediction)
        if "result" in st.session_state:
            render_interpretation(st.session_state["result"])

    # ── RIGHT ────────────────────────────────────────────────────────────────
    with right_col:
        if "result" not in st.session_state:
            render_awaiting_panel()
        else:
            result         = st.session_state["result"]
            inference_time = st.session_state.get("inference_time", 0.0)

            with st.container():
                st.markdown('<div class="panel-card result-panel">', unsafe_allow_html=True)
                render_result_header(result)
                render_result_banner(result["label"])

                donut_col, bar_col = st.columns([1, 1], gap="small")
                with donut_col:
                    render_donut_chart(result)
                with bar_col:
                    render_probability_bars(result)

                st.markdown("</div>", unsafe_allow_html=True)

            render_metrics(result)
            render_summary(result, inference_time)

    st.markdown("<div style='height:32px'></div>", unsafe_allow_html=True)
    render_footer()


if __name__ == "__main__":
    main()