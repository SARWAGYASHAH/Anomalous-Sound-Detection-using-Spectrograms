const state = { dashboard: null, models: [], selectedFile: null };
const $ = (id) => document.getElementById(id);
const formatMetric = (value, digits = 4) => value == null ? "--" : Number(value).toFixed(digits);
const escapeHtml = (text) => String(text ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

function badge(label) {
  const normalized = String(label || "normal").toLowerCase();
  return `<span class="badge ${normalized}">${escapeHtml(normalized.replace("_", " "))}</span>`;
}

function showToast(message) {
  const toast = $("toast");
  toast.textContent = message;
  toast.classList.remove("hidden");
  clearTimeout(showToast.timer);
  showToast.timer = setTimeout(() => toast.classList.add("hidden"), 4200);
}

async function api(path, options) {
  const response = await fetch(path, options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.detail || `Request failed (${response.status})`);
  return data;
}

function selectView(viewName) {
  document.querySelectorAll(".view").forEach((view) => view.classList.toggle("active", view.id === viewName));
  document.querySelectorAll(".nav-item").forEach((item) => item.classList.toggle("active", item.dataset.view === viewName));
}

function renderTrend(rows) {
  const svg = $("trendChart");
  const width = 760, height = 240, pad = 28;
  if (!rows.length) {
    svg.innerHTML = `<text x="28" y="125" fill="#9a9da3" font-family="Inter" font-size="13">No saved score stream found.</text>`;
    return;
  }
  const values = rows.flatMap((row) => [row.score, row.threshold]);
  const min = Math.min(...values) * 0.92;
  const max = Math.max(...values) * 1.06;
  const range = Math.max(max - min, 0.00001);
  const x = (i) => pad + i * ((width - pad * 2) / Math.max(rows.length - 1, 1));
  const y = (value) => height - pad - ((value - min) / range) * (height - pad * 2);
  const scoreLine = rows.map((row, index) => `${index ? "L" : "M"} ${x(index).toFixed(1)} ${y(row.score).toFixed(1)}`).join(" ");
  const thresholdY = y(rows[0].threshold).toFixed(1);
  const grid = [0, .25, .5, .75, 1].map((step) => {
    const gy = pad + step * (height - pad * 2);
    return `<line x1="${pad}" x2="${width - pad}" y1="${gy}" y2="${gy}" stroke="#1b1c1e" />`;
  }).join("");
  svg.innerHTML = `${grid}
    <line x1="${pad}" x2="${width - pad}" y1="${thresholdY}" y2="${thresholdY}" stroke="#50d8e9" stroke-dasharray="6 6" opacity=".8"/>
    <path d="${scoreLine}" fill="none" stroke="#7a85ff" stroke-width="2.5"/>
    <path d="${scoreLine} L ${x(rows.length - 1)} ${height - pad} L ${pad} ${height - pad} Z" fill="rgba(122,133,255,.10)"/>
    <circle cx="${x(rows.length - 1)}" cy="${y(rows[rows.length - 1].score)}" r="4" fill="#7a85ff"/>
    <text x="${width - pad - 116}" y="${Number(thresholdY) - 8}" fill="#50d8e9" font-size="11" font-family="Inter">threshold</text>`;
}

function renderDashboard(data) {
  state.dashboard = data;
  $("kpiSamples").textContent = data.kpis.test_samples ?? "--";
  $("kpiNormal").textContent = data.kpis.normal ?? "--";
  $("kpiFollow").textContent = data.kpis.follow_up ?? "--";
  $("kpiAlert").textContent = data.kpis.alert ?? "--";
  $("sideModel").textContent = data.health.active_model ? data.health.active_model.split("/")[2] || "Ready" : "No model";
  $("sideFramework").textContent = data.health.framework;

  if (data.evaluation) {
    $("thresholdLabel").textContent = `Threshold ${formatMetric(data.evaluation.threshold.value, 6)}`;
  }
  renderTrend(data.trend || []);

  const rows = data.recent_predictions || [];
  $("predictionRows").innerHTML = rows.map((row) => `<tr>
      <td>${escapeHtml(row.audio_file)}</td>
      <td>${escapeHtml(row.model_version || "--")}</td>
      <td>${formatMetric(row.score, 6)}</td>
      <td>${formatMetric(row.threshold, 6)}</td>
      <td>${badge(row.severity)}</td>
    </tr>`).join("");
  $("predictionEmpty").style.display = rows.length ? "none" : "block";
}

function renderModels(models) {
  state.models = models;
  const selector = $("modelSelector");
  selector.innerHTML = models.map((model) => `<option value="${escapeHtml(model.version)}">${escapeHtml(model.version)} / ${escapeHtml(model.model_file)}</option>`).join("");
  const evaluatedPath = state.dashboard?.evaluation?.model_path || "";
  const preferred = models.find((model) => model.model_path.replaceAll("\\", "/") === evaluatedPath.replaceAll("\\", "/"));
  if (preferred) selector.value = preferred.version;
  $("analysisModel").textContent = selector.value || "--";
  const activeVersion = state.dashboard?.active_model?.version;
  $("artifactRows").innerHTML = models.map((model) => `<tr>
      <td><strong>${escapeHtml(model.version)}</strong></td>
      <td>${escapeHtml(model.model_file)}</td>
      <td>${escapeHtml(model.files.join(" / "))}</td>
      <td>${new Date(model.modified_at).toLocaleDateString()}</td>
      <td>${model.version === activeVersion ? '<span class="meta">Active</span>' : '<span class="meta">Available</span>'}</td>
    </tr>`).join("");
}

async function loadEvaluation(split = "source_test") {
  try {
    const result = await api(`/api/evaluations/${split}`);
    $("metricAuc").textContent = formatMetric(result.metrics.auc_roc);
    $("metricPauc").textContent = formatMetric(result.metrics.pauc);
    $("metricPrecision").textContent = formatMetric(result.metrics.precision);
    $("metricRecall").textContent = formatMetric(result.metrics.recall);
    $("metricF1").textContent = formatMetric(result.metrics.f1);
    $("rocPlot").src = `${result.plot_urls.roc}?t=${Date.now()}`;
    $("scorePlot").src = `${result.plot_urls.scores}?t=${Date.now()}`;
    $("policyText").textContent = `Threshold ${formatMetric(result.threshold.value, 8)} is fitted from normal training samples using the ${result.threshold.method} method. This avoids using anomaly labels when establishing the decision boundary.`;
  } catch (error) {
    $("policyText").textContent = error.message;
    showToast(error.message);
  }
}

function setFile(file) {
  if (!file) return;
  if (!file.name.toLowerCase().endsWith(".wav")) {
    showToast("Only .wav files can be analyzed.");
    return;
  }
  state.selectedFile = file;
  $("selectedFile").textContent = `${file.name} - ${(file.size / 1024).toFixed(1)} KB ready for analysis`;
  $("analyzeButton").disabled = false;
}

async function analyzeSelectedFile() {
  if (!state.selectedFile) return;
  const button = $("analyzeButton");
  button.disabled = true;
  button.textContent = "Analyzing machine sound...";
  try {
    const modelVersion = $("modelSelector").value || "latest";
    const result = await api(`/api/predict?filename=${encodeURIComponent(state.selectedFile.name)}&model_version=${encodeURIComponent(modelVersion)}`, {
      method: "POST",
      headers: { "Content-Type": "audio/wav" },
      body: state.selectedFile,
    });
    $("resultPlaceholder").classList.add("hidden");
    $("resultContent").classList.remove("hidden");
    $("resultFilename").textContent = result.audio_file;
    const status = $("resultBadge");
    status.className = `badge ${result.severity}`;
    status.textContent = result.severity.replace("_", " ");
    $("resultScore").textContent = formatMetric(result.score, 8);
    $("resultThreshold").textContent = formatMetric(result.threshold, 8);
    $("resultRelative").textContent = `${(result.normalized_score * 100).toFixed(1)}%`;
    $("riskFill").style.width = `${Math.min(result.normalized_score * 100, 100)}%`;
    $("riskFill").style.background = result.severity === "alert" ? "#ff8178" : result.severity === "follow_up" ? "#ffb689" : "#45d39a";
    $("resultNote").textContent = result.severity === "alert"
      ? "Immediate maintenance review recommended. This signal falls inside the high-attention operating band."
      : result.severity === "follow_up"
        ? "Schedule a follow-up inspection and compare against recent gearbox recordings."
        : "No action required. The reconstruction pattern remains inside the normal operating band.";
    showToast("Audio analysis completed.");
    const dashboard = await api("/api/dashboard");
    renderDashboard(dashboard);
  } catch (error) {
    showToast(error.message);
  } finally {
    button.disabled = false;
    button.textContent = "Run anomaly analysis";
  }
}

async function initialize() {
  try {
    const [dashboard, models] = await Promise.all([api("/api/dashboard"), api("/api/models")]);
    renderDashboard(dashboard);
    renderModels(models);
    await loadEvaluation();
  } catch (error) {
    showToast(`Dashboard unavailable: ${error.message}`);
  }
}

document.querySelectorAll(".nav-item").forEach((button) => button.addEventListener("click", () => selectView(button.dataset.view)));
document.querySelectorAll(".jump-analyze").forEach((button) => button.addEventListener("click", () => selectView("analyze")));
$("modelSelector").addEventListener("change", (event) => { $("analysisModel").textContent = event.target.value; });
$("splitSelector").addEventListener("change", (event) => loadEvaluation(event.target.value));
$("audioFile").addEventListener("change", (event) => setFile(event.target.files[0]));
$("analyzeButton").addEventListener("click", analyzeSelectedFile);
["dragenter", "dragover"].forEach((name) => $("dropzone").addEventListener(name, (event) => { event.preventDefault(); $("dropzone").classList.add("dragging"); }));
["dragleave", "drop"].forEach((name) => $("dropzone").addEventListener(name, (event) => { event.preventDefault(); $("dropzone").classList.remove("dragging"); }));
$("dropzone").addEventListener("drop", (event) => setFile(event.dataTransfer.files[0]));
initialize();
