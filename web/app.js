const textInput = document.querySelector("#textInput");
const analyzeButton = document.querySelector("#analyzeButton");
const clearButton = document.querySelector("#clearButton");
const wordCount = document.querySelector("#wordCount");
const charCount = document.querySelector("#charCount");
const providerStatus = document.querySelector("#providerStatus");
const prediction = document.querySelector("#prediction");
const resultSubtitle = document.querySelector("#resultSubtitle");
const scoreValue = document.querySelector("#scoreValue");
const confidenceValue = document.querySelector("#confidenceValue");
const thresholdValue = document.querySelector("#thresholdValue");
const runtimeValue = document.querySelector("#runtimeValue");
const breakdown = document.querySelector("#breakdown");
const notes = document.querySelector("#notes");
const canvas = document.querySelector("#signalCanvas");
const ctx = canvas.getContext("2d");

const sampleText = `A public detector should treat AI authorship as a probability rather than a verdict. The Text-DIRE method masks parts of an input passage, asks a diffusion language model to reconstruct the missing tokens, and compares the reconstruction against the original. Text that falls closer to the model's learned writing manifold should produce a different reconstruction profile than text that falls farther away.`;

textInput.value = sampleText;

function modeValue() {
  return document.querySelector("input[name='mode']:checked").value;
}

function updateStats() {
  const words = textInput.value.trim().match(/\S+/g) || [];
  wordCount.textContent = `${words.length} words`;
  charCount.textContent = `${textInput.value.length} characters`;
}

async function loadHealth() {
  try {
    const response = await fetch("/health");
    const data = await response.json();
    providerStatus.textContent = `${data.provider} provider`;
  } catch {
    providerStatus.textContent = "provider unavailable";
  }
}

function setLoading(isLoading) {
  analyzeButton.disabled = isLoading;
  analyzeButton.textContent = isLoading ? "Analyzing" : "Analyze";
}

function renderEmptySignal() {
  const width = canvas.width;
  const height = canvas.height;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#f9faf8";
  ctx.fillRect(0, 0, width, height);
  drawGrid();
  drawThreshold(0.5);
}

function drawGrid() {
  ctx.strokeStyle = "#dce4e3";
  ctx.lineWidth = 1;
  for (let x = 60; x < canvas.width; x += 90) {
    ctx.beginPath();
    ctx.moveTo(x, 18);
    ctx.lineTo(x, canvas.height - 26);
    ctx.stroke();
  }
  for (let y = 38; y < canvas.height - 20; y += 40) {
    ctx.beginPath();
    ctx.moveTo(20, y);
    ctx.lineTo(canvas.width - 20, y);
    ctx.stroke();
  }
}

function drawThreshold(threshold) {
  const y = canvas.height - 24 - threshold * (canvas.height - 52);
  ctx.strokeStyle = "#a36b00";
  ctx.setLineDash([7, 7]);
  ctx.beginPath();
  ctx.moveTo(24, y);
  ctx.lineTo(canvas.width - 24, y);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = "#7b560e";
  ctx.font = "14px system-ui";
  ctx.fillText("threshold", 28, y - 8);
}

function renderSignal(result) {
  renderEmptySignal();
  const points = result.breakdown.map((item, index) => {
    const x = 70 + index * ((canvas.width - 140) / Math.max(1, result.breakdown.length - 1));
    const y = canvas.height - 24 - item.reconstruction_error * (canvas.height - 52);
    return { x, y, value: item.reconstruction_error, ratio: item.mask_ratio };
  });

  if (points.length > 1) {
    ctx.strokeStyle = "#0f766e";
    ctx.lineWidth = 4;
    ctx.beginPath();
    points.forEach((point, index) => {
      if (index === 0) ctx.moveTo(point.x, point.y);
      else ctx.lineTo(point.x, point.y);
    });
    ctx.stroke();
  }

  points.forEach((point) => {
    ctx.fillStyle = "#cf4f32";
    ctx.beginPath();
    ctx.arc(point.x, point.y, 7, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = "#151922";
    ctx.font = "14px system-ui";
    ctx.fillText(`m=${point.ratio}`, point.x - 20, canvas.height - 8);
  });
}

function renderResult(result) {
  const labelMap = {
    ai: "Likely AI-generated",
    human: "Likely human-written",
    uncertain: "Uncertain signal",
  };

  prediction.textContent = labelMap[result.prediction];
  prediction.className = `prediction tone-${result.prediction}`;
  resultSubtitle.textContent = `${result.provider} scoring completed`;
  scoreValue.textContent = result.score.toFixed(3);
  confidenceValue.textContent = `${Math.round(result.confidence * 100)}%`;
  thresholdValue.textContent = result.threshold.toFixed(2);
  runtimeValue.textContent = `${result.elapsed_seconds.toFixed(2)}s`;

  breakdown.replaceChildren(
    ...result.breakdown.map((item) => {
      const row = document.createElement("div");
      row.className = "bar-row";
      row.innerHTML = `
        <span>mask ${item.mask_ratio}</span>
        <span class="bar-track"><span class="bar-fill" style="width: ${Math.round(item.reconstruction_error * 100)}%"></span></span>
        <span>${item.reconstruction_error.toFixed(3)}</span>
      `;
      return row;
    }),
  );

  notes.replaceChildren(
    ...result.notes.map((note) => {
      const item = document.createElement("div");
      item.className = "note";
      item.textContent = note;
      return item;
    }),
  );

  renderSignal(result);
}

function renderError(message) {
  prediction.textContent = "Could not analyze text";
  prediction.className = "prediction tone-uncertain";
  resultSubtitle.textContent = message;
  notes.replaceChildren();
  breakdown.replaceChildren();
  scoreValue.textContent = "--";
  confidenceValue.textContent = "--";
  thresholdValue.textContent = "--";
  runtimeValue.textContent = "--";
  renderEmptySignal();
}

async function analyze() {
  const text = textInput.value.trim();
  if (!text) {
    renderError("Paste text before running the detector.");
    return;
  }

  setLoading(true);
  try {
    const response = await fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, mode: modeValue() }),
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "Analysis failed.");
    }
    renderResult(data);
  } catch (error) {
    renderError(error.message);
  } finally {
    setLoading(false);
  }
}

textInput.addEventListener("input", updateStats);
analyzeButton.addEventListener("click", analyze);
clearButton.addEventListener("click", () => {
  textInput.value = "";
  updateStats();
  textInput.focus();
});

updateStats();
renderEmptySignal();
loadHealth();
