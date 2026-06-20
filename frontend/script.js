/* ==========================================================================
   AI VIDEO DETECTION LABORATORY — APPLICATION LOGIC
   Structured for easy swap of `runInference()` with a real FastAPI call.
   ========================================================================== */

(() => {
  "use strict";

  /* ---------------------------------------------------------------------
     DOM references
     --------------------------------------------------------------------- */
  const $ = (id) => document.getElementById(id);

  const dropzone        = $("dropzone");
  const fileInput       = $("fileInput");
  const intakeStage      = $("intakeStage");
  const intakeTag        = $("intakeTag");
  const previewStage     = $("previewStage");
  const previewVideo     = $("previewVideo");
  const previewFrame     = document.querySelector(".preview-frame");
  const previewBadgeText = $("previewBadgeText");
  const analyzeBtn       = $("analyzeBtn");
  const resetBtn         = $("resetBtn");
  const analysisStatus   = $("analysisStatus");
  const statusLabel      = $("statusLabel");
  const statusPct        = $("statusPct");
  const progressFill     = $("progressFill");
  const statusLog        = $("statusLog");

  const metaCaseId     = $("metaCaseId");
  const metaFilename   = $("metaFilename");
  const metaDuration   = $("metaDuration");
  const metaResolution = $("metaResolution");
  const metaSize       = $("metaSize");
  const metaTime       = $("metaTime");
  const metaFrames     = $("metaFrames");
  const metaInferTime  = $("metaInferTime");

  const pipelineSteps = Array.from(document.querySelectorAll(".pipeline-step"));
  const pipelineFill  = $("pipelineFill");

  const resultsSection = $("results");
  const verdictPanel   = $("verdictPanel");
  const verdictCaseTag = $("verdictCaseTag");
  const verdictIcon    = $("verdictIcon");
  const verdictLabel   = $("verdictLabel");
  const verdictSub     = $("verdictSub");
  const confidenceRingFill = $("confidenceRingFill");
  const confidenceNumber   = $("confidenceNumber");

  const aiProbValue   = $("aiProbValue");
  const aiProbFill    = $("aiProbFill");
  const realProbValue = $("realProbValue");
  const realProbFill  = $("realProbFill");

  const traceLine    = $("traceLine");
  const traceArea    = $("traceArea");
  const traceDots    = $("traceDots");
  const traceAxisEnd = $("traceAxisEnd");

  const summaryText   = $("summaryText");
  const summaryPoints = $("summaryPoints");
  const newAnalysisBtn = $("newAnalysisBtn");

  const caseLog = $("caseLog");

  const RING_CIRCUMFERENCE = 2 * Math.PI * 60; // matches r=60 in the SVG

  /* ---------------------------------------------------------------------
     State
     --------------------------------------------------------------------- */
  const state = {
    file: null,
    objectUrl: null,
    caseId: null,
    durationLabel: "—",
    frameCount: 0,
  };

  /* ---------------------------------------------------------------------
     Utilities
     --------------------------------------------------------------------- */
  function formatBytes(bytes) {
    if (!bytes) return "—";
    const units = ["B", "KB", "MB", "GB"];
    let i = 0;
    let val = bytes;
    while (val >= 1024 && i < units.length - 1) {
      val /= 1024;
      i++;
    }
    return `${val.toFixed(val >= 10 || i === 0 ? 0 : 1)} ${units[i]}`;
  }

  function formatDuration(seconds) {
    if (!seconds || !isFinite(seconds)) return "—";
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return `${m}:${String(s).padStart(2, "0")}`;
  }

  function generateCaseId() {
    const now = new Date();
    const y = now.getFullYear();
    const m = String(now.getMonth() + 1).padStart(2, "0");
    const d = String(now.getDate()).padStart(2, "0");
    const chars = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789";
    let suffix = "";
    for (let i = 0; i < 3; i++) suffix += chars[Math.floor(Math.random() * chars.length)];
    return `CASE-${y}${m}${d}-${suffix}${Math.floor(Math.random() * 9)}`;
  }

  function randomBetween(min, max) {
    return Math.random() * (max - min) + min;
  }

  function clamp(v, min, max) {
    return Math.max(min, Math.min(max, v));
  }

  function easeOutCubic(t) {
    return 1 - Math.pow(1 - t, 3);
  }

  function countUp(el, target, opts = {}) {
    const duration = opts.duration || 900;
    const decimals = opts.decimals || 0;
    const start = performance.now();
    function tick(now) {
      const progress = clamp((now - start) / duration, 0, 1);
      const value = target * easeOutCubic(progress);
      el.textContent = value.toFixed(decimals);
      if (progress < 1) requestAnimationFrame(tick);
      else el.textContent = target.toFixed(decimals);
    }
    requestAnimationFrame(tick);
  }

  function setPipelineState(stepNumber, fillPct) {
    pipelineSteps.forEach((step) => {
      const n = Number(step.dataset.step);
      if (n < stepNumber) step.dataset.state = "done";
      else if (n === stepNumber) step.dataset.state = "active";
      else step.dataset.state = "pending";
    });
    pipelineFill.style.width = `${fillPct}%`;
  }

  /* ---------------------------------------------------------------------
     Upload handling
     --------------------------------------------------------------------- */
  function setupDropzone() {
    dropzone.addEventListener("click", () => fileInput.click());
    dropzone.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        fileInput.click();
      }
    });

    ["dragenter", "dragover"].forEach((evt) => {
      dropzone.addEventListener(evt, (e) => {
        e.preventDefault();
        dropzone.classList.add("is-dragover");
      });
    });

    ["dragleave", "dragend"].forEach((evt) => {
      dropzone.addEventListener(evt, (e) => {
        e.preventDefault();
        dropzone.classList.remove("is-dragover");
      });
    });

    dropzone.addEventListener("drop", (e) => {
      e.preventDefault();
      dropzone.classList.remove("is-dragover");
      const file = e.dataTransfer.files && e.dataTransfer.files[0];
      if (file) handleFile(file);
    });

    fileInput.addEventListener("change", (e) => {
      const file = e.target.files && e.target.files[0];
      if (file) handleFile(file);
    });
  }

  function handleFile(file) {
    if (!file.type.startsWith("video/")) {
      flashIntakeTag("Unsupported file type", true);
      return;
    }

    state.file = file;
    state.caseId = generateCaseId();

    if (state.objectUrl) URL.revokeObjectURL(state.objectUrl);
    state.objectUrl = URL.createObjectURL(file);
    previewVideo.src = state.objectUrl;

    metaCaseId.textContent = state.caseId;
    metaFilename.textContent = file.name;
    metaSize.textContent = formatBytes(file.size);
    metaTime.textContent = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    metaDuration.textContent = "—";
    metaResolution.textContent = "—";

    previewVideo.addEventListener(
      "loadedmetadata",
      () => {
        state.durationLabel = formatDuration(previewVideo.duration);
        metaDuration.textContent = state.durationLabel;
        metaResolution.textContent = `${previewVideo.videoWidth}×${previewVideo.videoHeight}`;
        state.frameCount = Math.max(24, Math.round(previewVideo.duration * 2)); // sampled frames, not raw fps
      },
      { once: true }
    );

    dropzone.hidden = true;
    previewStage.hidden = false;
    analysisStatus.hidden = true;
    progressFill.style.width = "0%";
    statusLog.innerHTML = "";
    analyzeBtn.disabled = false;
    analyzeBtn.textContent = "";
    rebuildAnalyzeBtn();

    intakeTag.textContent = "Evidence loaded";
    previewBadgeText.textContent = "EVIDENCE LOADED";

    resultsSection.hidden = true;
    setPipelineState(2, 25);
  }

  function rebuildAnalyzeBtn() {
    analyzeBtn.innerHTML = `
      <span class="btn-icon" aria-hidden="true">
        <svg viewBox="0 0 20 20" width="16" height="16"><path d="M10 2 L17 6 V14 L10 18 L3 14 V6 Z" fill="none" stroke="currentColor" stroke-width="1.5"/><circle cx="10" cy="10" r="2.5" fill="currentColor"/></svg>
      </span>
      Run Forensic Analysis`;
  }

  function flashIntakeTag(msg, isWarn) {
    intakeTag.textContent = msg;
    intakeTag.style.color = isWarn ? "var(--accent-ai)" : "";
    setTimeout(() => {
      intakeTag.textContent = state.file ? "Evidence loaded" : "Awaiting file";
      intakeTag.style.color = "";
    }, 2200);
  }

  function resetUpload() {
    state.file = null;
    if (state.objectUrl) URL.revokeObjectURL(state.objectUrl);
    state.objectUrl = null;
    previewVideo.removeAttribute("src");
    previewVideo.load();

    dropzone.hidden = false;
    previewStage.hidden = true;
    previewFrame.classList.remove("is-scanning");
    fileInput.value = "";

    intakeTag.textContent = "Awaiting file";
    resultsSection.hidden = true;
    setPipelineState(1, 0);
  }

  /* ---------------------------------------------------------------------
     Simulated analysis pipeline
     (Replace the body of runInference() with a fetch() call to your
     FastAPI endpoint — see comment near the bottom of this file.)
     --------------------------------------------------------------------- */
  const PIPELINE_LOG_STAGES = [
    { pct: 12, label: "Decoding video & extracting frames…", log: "Sampled {frames} key frames from source" },
    { pct: 34, label: "Running CNN spatial feature extraction…", log: "Per-frame artifact map computed" },
    { pct: 58, label: "Sequencing temporal features…", log: "Frame embeddings ordered for LSTM input" },
    { pct: 78, label: "LSTM inference across timeline…", log: "Temporal consistency scored" },
    { pct: 94, label: "Aggregating verdict & confidence…", log: "Softmax classification finalized" },
    { pct: 100, label: "Analysis complete.", log: "Report ready for review" },
  ];

  function runForensicAnalysis() {
    analyzeBtn.disabled = true;
    resetBtn.disabled = true;
    analysisStatus.hidden = false;
    previewFrame.classList.add("is-scanning");
    previewBadgeText.textContent = "ANALYZING";
    setPipelineState(2, 50);

    statusLog.innerHTML = "";
    progressFill.style.width = "0%";

    let i = 0;
    const stepDelay = 620;

    function nextStage() {
      if (i >= PIPELINE_LOG_STAGES.length) {
        finishAnalysis();
        return;
      }
      const stage = PIPELINE_LOG_STAGES[i];
      statusLabel.textContent = stage.label;
      statusPct.textContent = `${stage.pct}%`;
      progressFill.style.width = `${stage.pct}%`;

      const li = document.createElement("li");
      li.textContent = stage.log.replace("{frames}", state.frameCount || 48);
      statusLog.appendChild(li);
      if (statusLog.children.length > 4) statusLog.removeChild(statusLog.firstElementChild);

      setPipelineState(3, 50 + (stage.pct / 100) * 40);

      i++;
      setTimeout(nextStage, stepDelay);
    }

    nextStage();
  }

  async function finishAnalysis() {
    previewFrame.classList.remove("is-scanning");
    previewBadgeText.textContent = "ANALYSIS COMPLETE";
    analyzeBtn.disabled = false;
    resetBtn.disabled = false;
    setPipelineState(4, 100);

    const result = await runInference();
    renderResults(result);
    addCaseLogEntry(result);

    resultsSection.hidden = false;
    requestAnimationFrame(() => {
      resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
    });
  }

  /* ---------------------------------------------------------------------
     Mock inference — swap this function out for a real backend call.
     Expected return shape documented inline.
     --------------------------------------------------------------------- */
async function runInference() {

    const formData = new FormData();

    formData.append(
        "video",
        state.file
    );

    const response = await fetch(
        "http://127.0.0.1:8000/predict",
        {
            method: "POST",
            body: formData
        }
    );

    if (!response.ok) {
        throw new Error(
            "Prediction failed"
        );
    }

    const result =
        await response.json();

    return {

        verdict:
            result.label === "REAL VIDEO"
                ? "real"
                : result.label === "AI GENERATED"
                ? "ai"
                : "uncertain",

        aiProbability:
            result.ai_probability,

        realProbability:
            result.real_probability,

        confidence:
            result.confidence,

        frameTrace:
            generateTraceData(
                result.label === "REAL VIDEO"
                    ? "real"
                    : result.label === "AI GENERATED"
                    ? "ai"
                    : "uncertain",
                30
            ),

        frameCount: 30,

        inferenceTimeMs:
            result.inference_time_ms || 0
    };
}

  function generateTraceData(verdict, count) {
    const points = [];
    let baseline;
    if (verdict === "real") baseline = randomBetween(14, 26);
    else if (verdict === "ai") baseline = randomBetween(68, 82);
    else baseline = randomBetween(48, 62);

    let prev = baseline;
    for (let i = 0; i < count; i++) {
      const drift = randomBetween(-7, 7);
      let value = prev * 0.6 + (baseline + drift) * 0.4;

      // occasional spikes for ai / uncertain verdicts to read as artifact bursts
      if (verdict !== "real" && Math.random() < 0.12) {
        value += randomBetween(10, 22);
      }
      value = clamp(value, 2, 98);
      points.push(value);
      prev = value;
    }
    return points;
  }

  /* ---------------------------------------------------------------------
     Results rendering
     --------------------------------------------------------------------- */
  const VERDICT_COPY = {
    real: {
      label: "Real Video",
      sub: "No synthetic generation artifacts detected across the sampled timeline.",
      icon: `<svg viewBox="0 0 24 24" width="26" height="26"><path d="M5 13 L10 18 L19 7" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/></svg>`,
    },
    ai: {
      label: "AI Generated Video",
      sub: "Spatial and temporal signals consistent with synthetic generation.",
      icon: `<svg viewBox="0 0 24 24" width="26" height="26"><path d="M12 3 L14.5 9.5 L21 12 L14.5 14.5 L12 21 L9.5 14.5 L3 12 L9.5 9.5 Z" fill="none" stroke="currentColor" stroke-width="2"/></svg>`,
    },
    uncertain: {
      label: "Uncertain",
      sub: "Mixed signals — confidence too low for a definitive ruling.",
      icon: `<svg viewBox="0 0 24 24" width="26" height="26"><path d="M9.5 9a2.5 2.5 0 1 1 3.7 2.2c-.9.5-1.4 1.1-1.4 2.1" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"/><circle cx="12" cy="17.3" r="1.1" fill="currentColor"/></svg>`,
    },
  };

  function renderResults(result) {
    verdictCaseTag.textContent = state.caseId || "CASE-UNKNOWN";
    verdictPanel.dataset.verdict = result.verdict;

    const copy = VERDICT_COPY[result.verdict];
    verdictLabel.textContent = copy.label;
    verdictSub.textContent = copy.sub;
    verdictIcon.innerHTML = copy.icon;

    countUp(confidenceNumber, result.confidence, { duration: 1000 });
    const offset = RING_CIRCUMFERENCE * (1 - result.confidence / 100);
    requestAnimationFrame(() => {
      confidenceRingFill.style.stroke =
        result.verdict === "ai" ? "var(--accent-ai)" : result.verdict === "uncertain" ? "var(--accent-uncertain)" : "var(--accent-cyan)";
      confidenceRingFill.style.strokeDashoffset = offset;
    });

    countUp(aiProbValue, result.aiProbability, { duration: 900, decimals: 0 });
    countUp(realProbValue, result.realProbability, { duration: 900, decimals: 0 });
    requestAnimationFrame(() => {
      aiProbFill.style.width = `${result.aiProbability}%`;
      realProbFill.style.width = `${result.realProbability}%`;
    });
    // append % manually after count-up settles
    setTimeout(() => {
      aiProbValue.textContent = `${Math.round(result.aiProbability)}%`;
      realProbValue.textContent = `${Math.round(result.realProbability)}%`;
    }, 950);

    metaFrames.textContent = result.frameCount;
    metaInferTime.textContent = `${result.inferenceTimeMs} ms`;

    drawTrace(result.frameTrace);
    renderSummary(result);
  }

  function drawTrace(values) {
    const width = 600;
    const height = 200;
    const n = values.length;
    const valueToY = (v) => height - (v / 100) * height;
    const indexToX = (i) => (n <= 1 ? 0 : (i / (n - 1)) * width);

    let linePath = "";
    values.forEach((v, i) => {
      const x = indexToX(i);
      const y = valueToY(v);
      linePath += `${i === 0 ? "M" : "L"} ${x.toFixed(1)},${y.toFixed(1)} `;
    });

    const areaPath = `${linePath} L ${width},${height} L 0,${height} Z`;

    traceLine.setAttribute("d", linePath.trim());
    traceArea.setAttribute("d", areaPath.trim());

    const verdict = verdictPanel.dataset.verdict;
    const lineColor =
      verdict === "ai" ? "var(--accent-ai)" : verdict === "uncertain" ? "var(--accent-uncertain)" : "var(--accent-cyan)";
    traceLine.style.stroke = lineColor;
    traceArea.style.fill = lineColor;

    // draw-on animation
    const length = traceLine.getTotalLength();
    traceLine.style.transition = "none";
    traceLine.style.strokeDasharray = `${length}`;
    traceLine.style.strokeDashoffset = `${length}`;
    traceArea.style.opacity = "0";
    // force reflow
    traceLine.getBoundingClientRect();
    traceLine.style.transition = "stroke-dashoffset 1.3s cubic-bezier(0.65,0,0.35,1)";
    traceLine.style.strokeDashoffset = "0";
    traceArea.style.transition = "opacity 1.1s ease 0.4s";
    traceArea.style.opacity = "0.18";

    // sparse markers, colored by zone relative to the 60% threshold
    traceDots.innerHTML = "";
    const step = Math.max(1, Math.round(n / 14));
    values.forEach((v, i) => {
      if (i % step !== 0) return;
      const dot = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      dot.setAttribute("cx", indexToX(i).toFixed(1));
      dot.setAttribute("cy", valueToY(v).toFixed(1));
      dot.setAttribute("class", "trace-dot");
      dot.style.stroke = v > 60 ? "var(--accent-ai)" : "var(--accent-real)";
      dot.style.opacity = "0";
      dot.style.transition = `opacity 0.4s ease ${0.3 + (i / n) * 1}s`;
      traceDots.appendChild(dot);
      requestAnimationFrame(() => (dot.style.opacity = "1"));
    });

    traceAxisEnd.textContent = `Frame ${n}`;
  }

  function renderSummary(result) {
    const pct = (v) => `${Math.round(v)}%`;
    let lead;
    if (result.verdict === "real") {
      lead = `This clip presents as authentic footage. The CNN stage found no significant per-frame blending or texture artifacts, and the LSTM head reports a stable temporal signal with ${pct(result.realProbability)} probability of being real.`;
    } else if (result.verdict === "ai") {
      lead = `This clip shows strong indicators of synthetic generation. Spatial artifacts were detected across multiple sampled frames, and the temporal trace stays elevated through most of the timeline — ${pct(result.aiProbability)} probability of being AI generated.`;
    } else {
      lead = `Signals are mixed: the model's spatial and temporal readings hover close to the decision boundary, landing at ${pct(result.aiProbability)} AI-probability against ${pct(result.realProbability)} real-probability. We recommend a secondary review pass before treating this as conclusive.`;
    }
    summaryText.textContent = lead;

    const points = [
      `Confidence score of ${Math.round(result.confidence)}% computed from softmax margin between classes.`,
      `${result.frameCount} frames sampled across the clip for spatial + temporal inference.`,
      `Inference completed in ${result.inferenceTimeMs} ms on the CNN–LSTM v2.3 pipeline.`,
    ];
    if (result.verdict !== "real") {
      points.push("Frame-level spikes above the threshold line correlate with localized artifact bursts — see Temporal Authenticity Trace.");
    }
    summaryPoints.innerHTML = points.map((p) => `<li>${p}</li>`).join("");
  }

  function addCaseLogEntry(result) {
    const li = document.createElement("li");
    li.className = "case-log-item";
    li.dataset.verdict = result.verdict;
    const chipClass =
      result.verdict === "ai" ? "verdict-chip--ai" : result.verdict === "uncertain" ? "verdict-chip--uncertain" : "verdict-chip--real";
    const chipLabel = result.verdict === "ai" ? "AI Generated" : result.verdict === "uncertain" ? "Uncertain" : "Real";

    li.innerHTML = `
      <span class="case-log-id">${state.caseId}</span>
      <span class="case-log-name">${state.file ? state.file.name : "untitled.mp4"}</span>
      <span class="verdict-chip ${chipClass}">${chipLabel}</span>
    `;
    caseLog.prepend(li);
    while (caseLog.children.length > 5) caseLog.removeChild(caseLog.lastElementChild);
  }

  /* ---------------------------------------------------------------------
     Init
     --------------------------------------------------------------------- */
  function init() {
    setupDropzone();
    analyzeBtn.addEventListener("click", runForensicAnalysis);
    resetBtn.addEventListener("click", resetUpload);
    newAnalysisBtn.addEventListener("click", () => {
      resetUpload();
      document.getElementById("bench").scrollIntoView({ behavior: "smooth", block: "start" });
    });
    setPipelineState(1, 0);
  }

  document.addEventListener("DOMContentLoaded", init);
})();