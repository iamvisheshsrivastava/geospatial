const input = document.querySelector("#image-input");
const form = document.querySelector("#upload-form");
const dropzone = document.querySelector("#dropzone");
const preview = document.querySelector("#preview");
const previewEmpty = document.querySelector("#preview-empty");
const message = document.querySelector("#form-message");
const button = document.querySelector("#predict-button");
const statusDot = document.querySelector("#status-dot");
const statusLabel = document.querySelector("#status-label");
const modelPath = document.querySelector("#model-path");
const predictedClass = document.querySelector("#predicted-class");
const confidenceLabel = document.querySelector("#confidence-label");
const confidenceFill = document.querySelector("#confidence-fill");
const probabilities = document.querySelector("#probabilities");

let selectedFile = null;

function setMessage(text, isError = false) {
  message.textContent = text;
  message.style.color = isError ? "#b24a38" : "#667062";
}

function setStatus(payload) {
  statusDot.classList.remove("ready", "missing");
  statusDot.classList.add(payload.model_loaded ? "ready" : "missing");
  statusLabel.textContent = payload.model_loaded
    ? `Model loaded with ${payload.classes.length} classes`
    : "Model checkpoint is not loaded";
  modelPath.textContent = `Model: ${payload.model_path}`;
}

function renderProbabilities(items) {
  probabilities.innerHTML = "";
  Object.entries(items)
    .sort((a, b) => b[1] - a[1])
    .forEach(([name, score]) => {
      const row = document.createElement("div");
      row.className = "probability-row";
      row.innerHTML = `
        <span>${name}</span>
        <span class="probability-bar"><span style="width: ${Math.round(score * 100)}%"></span></span>
        <strong>${Math.round(score * 100)}%</strong>
      `;
      probabilities.appendChild(row);
    });
}

function previewFile(file) {
  selectedFile = file;
  setMessage(file ? file.name : "");
  probabilities.innerHTML = "";
  predictedClass.textContent = "Ready to classify";
  confidenceLabel.textContent = "0%";
  confidenceFill.style.width = "0%";

  if (!file) {
    preview.style.display = "none";
    previewEmpty.style.display = "block";
    return;
  }

  if (file.type.startsWith("image/")) {
    preview.src = URL.createObjectURL(file);
    preview.style.display = "block";
    previewEmpty.style.display = "none";
  } else {
    preview.style.display = "none";
    previewEmpty.style.display = "block";
    previewEmpty.textContent = "GeoTIFF selected";
  }
}

async function refreshHealth() {
  try {
    const response = await fetch("/health");
    setStatus(await response.json());
  } catch {
    statusDot.classList.add("missing");
    statusLabel.textContent = "API is not reachable";
  }
}

async function runPrediction(event) {
  event.preventDefault();
  if (!selectedFile) {
    setMessage("Choose an image first.", true);
    return;
  }

  button.disabled = true;
  button.textContent = "Classifying...";
  setMessage("Sending image to the model...");

  const formData = new FormData();
  formData.append("file", selectedFile);

  try {
    const response = await fetch("/predict", {
      method: "POST",
      body: formData,
    });
    const payload = await response.json();

    if (!response.ok) {
      throw new Error(payload.detail || "Prediction failed.");
    }

    const confidence = Math.round(payload.confidence * 100);
    predictedClass.textContent = payload.predicted_class;
    confidenceLabel.textContent = `${confidence}%`;
    confidenceFill.style.width = `${confidence}%`;
    renderProbabilities(payload.probabilities);
    setMessage("Classification complete.");
  } catch (error) {
    setMessage(error.message, true);
    await refreshHealth();
  } finally {
    button.disabled = false;
    button.textContent = "Run Classification";
  }
}

input.addEventListener("change", () => previewFile(input.files?.[0]));
form.addEventListener("submit", runPrediction);

["dragenter", "dragover"].forEach((eventName) => {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.add("dragging");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.remove("dragging");
  });
});

dropzone.addEventListener("drop", (event) => {
  const file = event.dataTransfer.files?.[0];
  if (file) {
    input.files = event.dataTransfer.files;
    previewFile(file);
  }
});

refreshHealth();
