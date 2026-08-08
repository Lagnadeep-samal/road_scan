/* =========================================================================
   RoadScan front-end logic
   Talks to the FastAPI backend at /api/*. Same-origin by default; override
   API_BASE below if the frontend is ever hosted separately from the API.
   ========================================================================= */

const API_BASE = "";

const ALLOWED_TYPES = ["image/jpeg", "image/png", "image/webp", "image/bmp"];
const MAX_FILE_SIZE_MB = 15;

// ---- element refs ---------------------------------------------------------

const statusPill = document.getElementById("statusPill");
const statusText = document.getElementById("statusText");

const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("fileInput");
const dzEmpty = document.getElementById("dzEmpty");
const dzPreview = document.getElementById("dzPreview");
const previewImg = document.getElementById("previewImg");
const scanBeam = document.getElementById("scanBeam");
const dzClear = document.getElementById("dzClear");
const fileInfo = document.getElementById("fileInfo");
const fileNameEl = document.getElementById("fileName");
const fileSizeEl = document.getElementById("fileSize");
const analyzeBtn = document.getElementById("analyzeBtn");
const analyzeLabel = document.getElementById("analyzeLabel");

const resultsSection = document.getElementById("results");
const conditionHeadline = document.getElementById("conditionHeadline");
const metaTotal = document.getElementById("metaTotal");
const metaTime = document.getElementById("metaTime");
const toggleImgBtn = document.getElementById("toggleImgBtn");
const resultImg = document.getElementById("resultImg");

const countMinor = document.getElementById("countMinor");
const countMedium = document.getElementById("countMedium");
const countMajor = document.getElementById("countMajor");

const gaugeScore = document.getElementById("gaugeScore");
const gaugeMarker = document.getElementById("gaugeMarker");

const logList = document.getElementById("logList");
const logCount = document.getElementById("logCount");

const resetBtn = document.getElementById("resetBtn");
const toastContainer = document.getElementById("toastContainer");

// ---- state ----------------------------------------------------------------

let selectedFile = null;
let originalObjectUrl = null;
let showingOriginal = false;
let lastAnnotatedUrl = null;

// ---- toast helper -----------------------------------------------------

function showToast(message, type = "error", timeout = 5000) {
  const el = document.createElement("div");
  el.className = `toast toast-${type}`;
  el.textContent = message;
  toastContainer.appendChild(el);
  setTimeout(() => el.remove(), timeout);
}

// ---- backend health check ----------------------------------------------

async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE}/api/health`);
    if (!res.ok) throw new Error("bad status");
    const data = await res.json();
    if (data.model_loaded) {
      setStatus("online", "Model ready");
    } else {
      setStatus("degraded", data.error ? "Model unavailable" : "Degraded");
    }
  } catch (err) {
    setStatus("offline", "Backend unreachable");
  }
}

function setStatus(state, label) {
  statusPill.dataset.state = state;
  statusText.textContent = label;
}

// ---- file selection ---------------------------------------------------

function humanFileSize(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  const units = ["KB", "MB", "GB"];
  let i = -1;
  do {
    bytes /= 1024;
    i++;
  } while (bytes >= 1024 && i < units.length - 1);
  return `${bytes.toFixed(1)} ${units[i]}`;
}

function validateFile(file) {
  if (!ALLOWED_TYPES.includes(file.type)) {
    return "Please choose a JPG, PNG, WEBP, or BMP image.";
  }
  if (file.size > MAX_FILE_SIZE_MB * 1024 * 1024) {
    return `That file is too large. Max size is ${MAX_FILE_SIZE_MB}MB.`;
  }
  if (file.size === 0) {
    return "That file appears to be empty.";
  }
  return null;
}

function selectFile(file) {
  const error = validateFile(file);
  if (error) {
    showToast(error);
    return;
  }

  selectedFile = file;

  if (originalObjectUrl) URL.revokeObjectURL(originalObjectUrl);
  originalObjectUrl = URL.createObjectURL(file);

  previewImg.src = originalObjectUrl;
  dzEmpty.hidden = true;
  dzPreview.hidden = false;

  fileInfo.hidden = false;
  fileNameEl.textContent = file.name;
  fileSizeEl.textContent = humanFileSize(file.size);

  analyzeBtn.disabled = false;

  // collapse any previous report when a new photo is chosen
  resultsSection.hidden = true;
}

function clearFile() {
  selectedFile = null;
  if (originalObjectUrl) URL.revokeObjectURL(originalObjectUrl);
  originalObjectUrl = null;

  fileInput.value = "";
  dzEmpty.hidden = false;
  dzPreview.hidden = true;
  fileInfo.hidden = true;
  analyzeBtn.disabled = true;
  resultsSection.hidden = true;
}

// ---- dropzone interactions ----------------------------------------------

dropzone.addEventListener("click", () => fileInput.click());
dropzone.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") {
    e.preventDefault();
    fileInput.click();
  }
});

fileInput.addEventListener("change", () => {
  if (fileInput.files && fileInput.files[0]) selectFile(fileInput.files[0]);
});

["dragenter", "dragover"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.add("drag-over");
  })
);

["dragleave", "drop"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.remove("drag-over");
  })
);

dropzone.addEventListener("drop", (e) => {
  const file = e.dataTransfer.files && e.dataTransfer.files[0];
  if (file) selectFile(file);
});

dzClear.addEventListener("click", (e) => {
  e.stopPropagation();
  clearFile();
});

// ---- analyze ---------------------------------------------------------

analyzeBtn.addEventListener("click", async () => {
  if (!selectedFile) return;

  setAnalyzing(true);

  const formData = new FormData();
  formData.append("file", selectedFile);

  try {
    const res = await fetch(`${API_BASE}/api/detect`, {
      method: "POST",
      body: formData,
    });

    let payload;
    try {
      payload = await res.json();
    } catch {
      throw new Error("The server sent back something unexpected. Please try again.");
    }

    if (!res.ok) {
      throw new Error(payload.detail || `Request failed (${res.status}).`);
    }

    renderResults(payload);
    showToast("Detection complete.", "success", 3000);
  } catch (err) {
    console.error(err);
    if (err instanceof TypeError) {
      showToast("Could not reach the backend. Check that the API server is running.");
    } else {
      showToast(err.message || "Something went wrong during detection.");
    }
  } finally {
    setAnalyzing(false);
  }
});

function setAnalyzing(isAnalyzing) {
  analyzeBtn.disabled = isAnalyzing;
  scanBeam.hidden = !isAnalyzing;
  analyzeLabel.textContent = isAnalyzing ? "Scanning…" : "Analyze Road";
  analyzeBtn.classList.toggle("loading", isAnalyzing);
}

// ---- render results ----------------------------------------------------

function animateCount(el, target) {
  const duration = 500;
  const start = performance.now();
  function tick(now) {
    const progress = Math.min((now - start) / duration, 1);
    el.textContent = Math.round(progress * target);
    if (progress < 1) requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
}

const SEVERITY_COLOR = {
  minor_pothole: "var(--minor)",
  medium_pothole: "var(--medium)",
  major_pothole: "var(--major)",
};

function renderResults(data) {
  resultsSection.hidden = false;

  conditionHeadline.textContent = `Road condition: ${data.road_condition.toUpperCase()}`;
  conditionHeadline.style.color =
    data.road_condition === "Good" ? "var(--minor)" :
    data.road_condition === "Moderate" ? "var(--medium)" : "var(--major)";

  metaTotal.textContent = `${data.total} pothole${data.total === 1 ? "" : "s"}`;
  metaTime.textContent = `${data.processing_time_ms.toFixed(0)}ms`;

  animateCount(countMinor, data.minor);
  animateCount(countMedium, data.medium);
  animateCount(countMajor, data.major);

  gaugeScore.textContent = data.road_score;
  const clamped = Math.min(data.road_score, 15);
  gaugeMarker.style.left = `${(clamped / 15) * 100}%`;

  lastAnnotatedUrl = data.image_url.startsWith("http")
    ? data.image_url
    : `${API_BASE}${data.image_url}`;
  resultImg.src = `${lastAnnotatedUrl}?t=${Date.now()}`;
  showingOriginal = false;
  toggleImgBtn.textContent = "show original";

  renderLog(data.detections);

  resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
}

function renderLog(detections) {
  logList.innerHTML = "";
  logCount.textContent = `${detections.length} ${detections.length === 1 ? "entry" : "entries"}`;

  if (detections.length === 0) {
    const empty = document.createElement("div");
    empty.className = "log-empty";
    empty.textContent = "No potholes crossed the confidence threshold for this photo.";
    logList.appendChild(empty);
    return;
  }

  detections
    .slice()
    .sort((a, b) => b.confidence - a.confidence)
    .forEach((d) => {
      const row = document.createElement("div");
      row.className = "log-row";

      const dot = document.createElement("span");
      dot.className = "dot";
      dot.style.background = SEVERITY_COLOR[d.class_name] || "var(--muted)";

      const sev = document.createElement("span");
      sev.className = "log-severity";
      sev.textContent = d.severity;

      const confWrap = document.createElement("div");
      const barTrack = document.createElement("div");
      barTrack.className = "log-conf-bar-track";
      const barFill = document.createElement("div");
      barFill.className = "log-conf-bar-fill";
      barFill.style.width = `${Math.round(d.confidence * 100)}%`;
      barFill.style.background = SEVERITY_COLOR[d.class_name] || "var(--muted)";
      barTrack.appendChild(barFill);
      confWrap.appendChild(barTrack);
      confWrap.style.display = "flex";
      confWrap.style.flexDirection = "column";
      confWrap.style.gap = "3px";
      const confLabel = document.createElement("span");
      confLabel.style.fontSize = "10px";
      confLabel.style.color = "var(--muted)";
      confLabel.textContent = `${Math.round(d.confidence * 100)}%`;
      confWrap.appendChild(confLabel);

      const coords = document.createElement("span");
      coords.className = "log-coords";
      coords.textContent = `${d.box[0]},${d.box[1]} → ${d.box[2]},${d.box[3]}`;

      row.appendChild(dot);
      row.appendChild(sev);
      row.appendChild(confWrap);
      row.appendChild(coords);
      logList.appendChild(row);
    });
}

toggleImgBtn.addEventListener("click", () => {
  showingOriginal = !showingOriginal;
  resultImg.src = showingOriginal ? originalObjectUrl : `${lastAnnotatedUrl}?t=${Date.now()}`;
  toggleImgBtn.textContent = showingOriginal ? "show detections" : "show original";
});

resetBtn.addEventListener("click", () => {
  clearFile();
  document.getElementById("hero").scrollIntoView({ behavior: "smooth" });
});

// ---- boot ---------------------------------------------------------------

checkHealth();
