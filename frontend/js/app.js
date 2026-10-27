/**
 * Product Review Sentiment Analysis - Frontend Application Logic
 * Integrates with Flask REST backend endpoints (/api/health, /api/predict, /api/metrics).
 */

const API_BASE_URL = window.location.origin.includes("localhost") || window.location.origin.includes("127.0.0.1")
  ? ""
  : "http://127.0.0.1:5000";

// State
let isModelAvailable = false;
let currentActiveSection = "home-section";

// DOM Elements
const navMenu = document.getElementById("navMenu");
const reviewInput = document.getElementById("reviewInput");
const charCounter = document.getElementById("charCounter");
const btnPredict = document.getElementById("btnPredict");
const btnClear = document.getElementById("btnClear");

// Result States
const resultPlaceholder = document.getElementById("resultPlaceholder");
const resultLoading = document.getElementById("resultLoading");
const resultError = document.getElementById("resultError");
const resultDisplay = document.getElementById("resultDisplay");
const errorTitle = document.getElementById("errorTitle");
const errorMessage = document.getElementById("errorMessage");

// Result Presentation Elements
const sentimentBadge = document.getElementById("sentimentBadge");
const sentimentIcon = document.getElementById("sentimentIcon");
const sentimentLabel = document.getElementById("sentimentLabel");
const confidenceValue = document.getElementById("confidenceValue");
const cleanedTextOutput = document.getElementById("cleanedTextOutput");
const probabilityBars = document.getElementById("probabilityBars");

// Health & Status Elements
const statusDot = document.getElementById("statusDot");
const statusText = document.getElementById("statusText");
const modelWarningBanner = document.getElementById("modelWarningBanner");
const modelBannerText = document.getElementById("modelBannerText");
const engineStatusVal = document.getElementById("engineStatusVal");
const engineStatusSub = document.getElementById("engineStatusSub");

/* ==========================================================================
   Navigation
   ========================================================================== */

function initNavigation() {
  const navButtons = document.querySelectorAll(".nav-item");
  navButtons.forEach(btn => {
    btn.addEventListener("click", () => {
      const targetId = btn.getAttribute("data-target");
      navigateToSection(targetId);
    });
  });
}

function navigateToSection(targetId) {
  // Update nav buttons
  document.querySelectorAll(".nav-item").forEach(btn => {
    if (btn.getAttribute("data-target") === targetId) {
      btn.classList.add("active");
    } else {
      btn.classList.remove("active");
    }
  });

  // Update sections
  document.querySelectorAll(".page-section").forEach(sec => {
    if (sec.id === targetId) {
      sec.classList.add("active");
    } else {
      sec.classList.remove("active");
    }
  });

  currentActiveSection = targetId;

  // Auto-fetch metrics if navigating to performance tab
  if (targetId === "performance-section") {
    fetchMetrics();
  }

  // Scroll to top
  window.scrollTo({ top: 0, behavior: "smooth" });
}

/* ==========================================================================
   Health Check & Backend Status
   ========================================================================== */

async function checkBackendHealth() {
  try {
    const res = await fetch(`${API_BASE_URL}/api/health`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    statusDot.className = "status-dot";

    if (data.model_loaded) {
      isModelAvailable = true;
      statusDot.classList.add("online");
      statusText.textContent = "Model Ready";
      modelWarningBanner.classList.add("hidden");
      engineStatusVal.textContent = "LSTM (Active)";
      engineStatusSub.textContent = `Supported classes: ${data.supported_classes.join(", ")}`;
    } else {
      isModelAvailable = false;
      statusDot.classList.add("offline");
      statusText.textContent = "Model Not Trained";
      modelWarningBanner.classList.remove("hidden");
      modelBannerText.textContent = data.message || "Model artifact not found. Run 'python -m src.train' to train.";
      engineStatusVal.textContent = "Ready to Train";
      engineStatusSub.textContent = "Dataset ready in dataset/ directory";
    }
  } catch (err) {
    statusDot.className = "status-dot offline";
    statusText.textContent = "Backend Offline";
    modelWarningBanner.classList.remove("hidden");
    modelBannerText.textContent = "Cannot connect to Flask server. Run 'python -m backend.app' to start backend.";
  }
}

/* ==========================================================================
   Live Sentiment Prediction
   ========================================================================== */

function initReviewInput() {
  reviewInput.addEventListener("input", () => {
    const len = reviewInput.value.length;
    charCounter.textContent = `${len} character${len === 1 ? "" : "s"}`;
  });

  // Support Ctrl+Enter for quick predict
  reviewInput.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
      handlePredictSentiment();
    }
  });
}

function loadExampleReview(text) {
  reviewInput.value = text;
  reviewInput.dispatchEvent(new Event("input"));
  navigateToSection("analyzer-section");
  reviewInput.focus();
}

function handleClearInput() {
  reviewInput.value = "";
  charCounter.textContent = "0 characters";
  showState("placeholder");
  reviewInput.focus();
}

function showState(stateName) {
  resultPlaceholder.classList.add("hidden");
  resultLoading.classList.add("hidden");
  resultError.classList.add("hidden");
  resultDisplay.classList.add("hidden");

  if (stateName === "placeholder") resultPlaceholder.classList.remove("hidden");
  else if (stateName === "loading") resultLoading.classList.remove("hidden");
  else if (stateName === "error") resultError.classList.remove("hidden");
  else if (stateName === "display") resultDisplay.classList.remove("hidden");
}

async function handlePredictSentiment() {
  const review = reviewInput.value.trim();

  if (!review) {
    showState("error");
    errorTitle.textContent = "Empty Input";
    errorMessage.textContent = "Please enter some review text before predicting sentiment.";
    reviewInput.focus();
    return;
  }

  showState("loading");
  btnPredict.disabled = true;

  try {
    const res = await fetch(`${API_BASE_URL}/api/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ review })
    });

    const data = await res.json();

    if (!res.ok) {
      showState("error");
      errorTitle.textContent = data.error || `Error ${res.status}`;
      errorMessage.textContent = data.details || data.remedy || "Inference failed.";
      return;
    }

    renderPrediction(data);
    showState("display");
  } catch (err) {
    showState("error");
    errorTitle.textContent = "Connection Failure";
    errorMessage.textContent = "Could not contact prediction API. Please verify that the Flask server is running.";
  } finally {
    btnPredict.disabled = false;
  }
}

function renderPrediction(data) {
  const sentiment = (data.predicted_sentiment || "neutral").toLowerCase();
  const confidence = data.confidence ? Math.round(data.confidence * 1000) / 10 : 0.0;

  // Sentiment badge
  sentimentBadge.className = `sentiment-badge ${sentiment}`;
  sentimentLabel.textContent = sentiment.charAt(0).toUpperCase() + sentiment.slice(1);

  if (sentiment === "positive") {
    sentimentIcon.textContent = "😄";
  } else if (sentiment === "negative") {
    sentimentIcon.textContent = "😞";
  } else {
    sentimentIcon.textContent = "😐";
  }

  // Confidence
  confidenceValue.textContent = `${confidence.toFixed(1)}%`;

  // Cleaned Text
  cleanedTextOutput.textContent = data.cleaned_review || "(Text was empty after normalization)";

  // Probability distribution bars
  probabilityBars.innerHTML = "";
  const probs = data.probabilities || {};

  const order = ["positive", "neutral", "negative"];
  const sortedClasses = Object.keys(probs).sort((a, b) => {
    const ia = order.indexOf(a.toLowerCase());
    const ib = order.indexOf(b.toLowerCase());
    if (ia !== -1 && ib !== -1) return ia - ib;
    return a.localeCompare(b);
  });

  sortedClasses.forEach(cls => {
    const probVal = probs[cls] || 0.0;
    const pct = (probVal * 100).toFixed(1);
    const clsLower = cls.toLowerCase();

    const barItem = document.createElement("div");
    barItem.className = "prob-bar-item";
    barItem.innerHTML = `
      <div class="prob-header">
        <span class="font-semibold text-capitalize">${cls}</span>
        <span class="font-mono">${pct}%</span>
      </div>
      <div class="prob-track">
        <div class="prob-fill ${clsLower}" style="width: ${pct}%"></div>
      </div>
    `;
    probabilityBars.appendChild(barItem);
  });
}

/* ==========================================================================
   Model Performance & Metrics Fetching
   ========================================================================== */

async function fetchMetrics(forceRefresh = false) {
  try {
    const res = await fetch(`${API_BASE_URL}/api/metrics`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    if (data.status === "available" && data.metrics) {
      renderMetrics(data.metrics, data.training_summary);
      if (forceRefresh) {
        refreshChartImages();
      }
    } else {
      renderMetricsUnavailable(data.message || "Metrics unavailable until model training is executed.");
    }
  } catch (err) {
    renderMetricsUnavailable("Could not connect to backend to retrieve metrics.");
  }
}

function renderMetrics(metrics, trainingSummary) {
  document.getElementById("metricAccuracy").textContent = `${(metrics.accuracy * 100).toFixed(2)}%`;
  document.getElementById("metricPrecision").textContent = `${(metrics.precision_macro * 100).toFixed(2)}%`;
  document.getElementById("metricRecall").textContent = `${(metrics.recall_macro * 100).toFixed(2)}%`;
  document.getElementById("metricF1").textContent = `${(metrics.f1_macro * 100).toFixed(2)}%`;

  // Render per-class table
  const tbody = document.getElementById("perClassTableBody");
  tbody.innerHTML = "";

  const perClass = metrics.per_class_metrics || {};
  Object.keys(perClass).forEach(clsName => {
    const item = perClass[clsName];
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><strong class="text-capitalize">${clsName}</strong></td>
      <td>${(item.precision * 100).toFixed(2)}%</td>
      <td>${(item.recall * 100).toFixed(2)}%</td>
      <td><strong>${(item.f1_score * 100).toFixed(2)}%</strong></td>
      <td>${item.support.toLocaleString()}</td>
    `;
    tbody.appendChild(tr);
  });
}

function renderMetricsUnavailable(msg) {
  document.getElementById("metricAccuracy").textContent = "--";
  document.getElementById("metricPrecision").textContent = "--";
  document.getElementById("metricRecall").textContent = "--";
  document.getElementById("metricF1").textContent = "--";

  const tbody = document.getElementById("perClassTableBody");
  tbody.innerHTML = `<tr><td colspan="5" class="text-center text-muted" style="padding: 2rem;">${msg}</td></tr>`;
}

function handleImageError(img, message) {
  const parent = img.parentElement;
  parent.innerHTML = `
    <div class="chart-fallback">
      <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" style="margin-bottom: 0.5rem; opacity: 0.5;">
        <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
        <circle cx="8.5" cy="8.5" r="1.5"></circle>
        <polyline points="21 15 16 10 5 21"></polyline>
      </svg>
      <p>${message}</p>
    </div>
  `;
}

function refreshChartImages() {
  const images = document.querySelectorAll(".chart-img");
  const t = Date.now();
  images.forEach(img => {
    const baseSrc = img.src.split("?")[0];
    img.src = `${baseSrc}?t=${t}`;
  });
}

/* ==========================================================================
   Initialization
   ========================================================================== */

document.addEventListener("DOMContentLoaded", () => {
  initNavigation();
  initReviewInput();
  checkBackendHealth();

  // Periodic health ping
  setInterval(checkBackendHealth, 15000);
});
