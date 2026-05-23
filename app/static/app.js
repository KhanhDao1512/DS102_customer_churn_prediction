const form = document.getElementById("churn-form");
const statusEl = document.getElementById("status");
const resultText = document.getElementById("result-text");
const resultScore = document.getElementById("result-score");
const riskBar = document.getElementById("risk-bar");
const probabilityText = document.getElementById("probability-text");
const sourceText = document.getElementById("source-text");
const modelSource = document.getElementById("model-source");
const predictionState = document.getElementById("prediction-state");
const predictionStateBadge = document.getElementById("prediction-state-badge");
const profileStatus = document.getElementById("profile-status");
const recommendation = document.getElementById("recommendation");
const actionButtons = document.querySelectorAll("[data-action]");

const demoProfile = {
  gender: "Female",
  SeniorCitizen: "0",
  Partner: "No",
  Dependents: "No",
  tenure: "8",
  PhoneService: "Yes",
  MultipleLines: "No",
  InternetService: "Fiber optic",
  OnlineSecurity: "No",
  OnlineBackup: "No",
  DeviceProtection: "No",
  TechSupport: "No",
  StreamingTV: "Yes",
  StreamingMovies: "Yes",
  Contract: "Month-to-month",
  PaperlessBilling: "Yes",
  PaymentMethod: "Electronic check",
  MonthlyCharges: "89.9",
  TotalCharges: "650.5",
};

function toPayload(formData) {
  const payload = {};

  for (const [key, value] of formData.entries()) {
    payload[key] =
      key === "SeniorCitizen" || key === "tenure"
        ? Number.parseInt(value, 10)
        : key === "MonthlyCharges" || key === "TotalCharges"
          ? Number.parseFloat(value)
          : value;
  }

  return payload;
}

function setFieldValue(name, value) {
  const field = form.elements.namedItem(name);
  if (field) {
    field.value = value;
  }
}

function getFilledRatio() {
  const requiredFields = Array.from(form.querySelectorAll("[required]"));
  const filledCount = requiredFields.filter((field) => String(field.value).trim() !== "").length;
  return Math.round((filledCount / requiredFields.length) * 100);
}

function updateProfileProgress() {
  profileStatus.textContent = `${getFilledRatio()}%`;
}

function getRecommendation(probability) {
  if (probability >= 0.75) {
    return "High risk. Route the account to retention review and check contract or support issues first.";
  }
  if (probability >= 0.5) {
    return "Moderate risk. A targeted follow-up or service offer could improve retention.";
  }
  return "Lower risk. Keep monitoring and use periodic retention nudges.";
}

function renderResult(data) {
  const probability = Number(data.probability ?? 0);
  const percentage = Math.round(probability * 1000) / 10;

  resultText.textContent = data.prediction;
  resultScore.textContent = `${percentage.toFixed(1)}%`;
  probabilityText.textContent = `${percentage.toFixed(1)}% probability`;
  sourceText.textContent = data.source === "model" ? "Backed by saved model" : "Preview logic";
  modelSource.textContent = data.source === "model" ? "Model" : "Preview";
  predictionState.textContent = data.prediction;
  predictionStateBadge.textContent = data.prediction;
  recommendation.textContent = getRecommendation(probability);
  riskBar.style.width = `${Math.min(Math.max(probability * 100, 0), 100)}%`;
}

function resetResult() {
  resultText.textContent = "Submit the form to see churn risk.";
  resultScore.textContent = "--";
  probabilityText.textContent = "0%";
  sourceText.textContent = "No request yet";
  modelSource.textContent = "Preview";
  predictionState.textContent = "Waiting";
  predictionStateBadge.textContent = "Waiting";
  recommendation.textContent = "Use the form to generate a recommendation.";
  riskBar.style.width = "0%";
  statusEl.textContent = "Ready.";
}

function fillDemoProfile() {
  Object.entries(demoProfile).forEach(([key, value]) => {
    setFieldValue(key, value);
  });
  updateProfileProgress();
  statusEl.textContent = "Demo profile loaded.";
}

async function copyResult() {
  const text = [
    `Prediction: ${resultText.textContent}`,
    `Risk: ${resultScore.textContent}`,
    `Source: ${sourceText.textContent}`,
    `Recommendation: ${recommendation.textContent}`,
  ].join("\n");

  try {
    await navigator.clipboard.writeText(text);
    statusEl.textContent = "Result copied to clipboard.";
  } catch (error) {
    statusEl.textContent = "Copy failed in this browser.";
  }
}

actionButtons.forEach((button) => {
  button.addEventListener("click", () => {
    const action = button.dataset.action;
    if (action === "load-demo") {
      fillDemoProfile();
    }
    if (action === "reset") {
      form.reset();
      updateProfileProgress();
      resetResult();
    }
    if (action === "copy") {
      copyResult();
    }
  });
});

form.addEventListener("input", updateProfileProgress);

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  statusEl.textContent = "Running prediction...";
  predictionState.textContent = "Running";
  predictionStateBadge.textContent = "Running";

  const payload = toPayload(new FormData(form));

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      throw new Error("Prediction request failed");
    }

    const data = await response.json();
    renderResult(data);
    statusEl.textContent = "Prediction complete.";
  } catch (error) {
    statusEl.textContent = "Prediction failed. Check the backend.";
    predictionState.textContent = "Error";
    predictionStateBadge.textContent = "Error";
  }
});

updateProfileProgress();
resetResult();
