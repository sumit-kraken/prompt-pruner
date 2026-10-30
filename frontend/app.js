// Prompt Pruning & LLM Acceleration Studio Client Logic
// Dual Runtime Support: Ollama (Backend SSE) + WebGPU (Client-side In-Browser)
// Phase 2 Integration: Query-Conditioned Pruning via WebGPU WGSL Compute Shader

import { WebGPUQueryPruner } from "./query_pruner_webgpu.js";

const API_BASE = window.location.port === "8000" 
  ? "" 
  : "http://localhost:8000";

// Preset diagnostic logs
const PRESET_LOGS = {
  auth: {
    title: "OAuth2 NullPointer",
    text: `2026-10-29 12:04:18.291 [pool-3-thread-7] ERROR com.cloudcorp.auth.service.OAuth2TokenValidator - Validation failure during bearer token introspection.
java.lang.NullPointerException: Cannot invoke "com.cloudcorp.session.UserSession.getTenantId()" because the return value of "com.cloudcorp.auth.cache.DistributedSessionStore.getActiveSession(String)" is null
    at com.cloudcorp.auth.service.OAuth2TokenValidator.validateSignature(OAuth2TokenValidator.java:184) ~[auth-service-2.14.0.jar:2.14.0]
    at com.cloudcorp.auth.service.OAuth2TokenValidator.introspectToken(OAuth2TokenValidator.java:92) ~[auth-service-2.14.0.jar:2.14.0]
    at com.cloudcorp.auth.controller.TokenValidationEndpoint.handleIntrospectRequest(TokenValidationEndpoint.java:47) ~[auth-service-2.14.0.jar:2.14.0]
    at com.cloudcorp.gateway.filter.SecurityContextFilter.doFilter(SecurityContextFilter.java:119) ~[gateway-core-1.8.2.jar:1.8.2]
[INFO] Connection to Redis cluster host 10.0.4.12:6379 failed after timeout of 3000ms. Retrying attempt 3 of 5.
[WARN] Circuit breaker 'auth-session-cache' transitioned from CLOSED to HALF_OPEN state due to high failure rate (78.4% errors over last 60s).
client_ip=192.168.1.104 client_id=mobile-client-v3 req_id=f7e8a91c-8b43-4122-bf89-219e917d12a9`,
    query: "What caused the authentication validation failure, which class and line threw NullPointerException, and what happened to the Redis connection?"
  },
  k8s: {
    title: "Kubernetes OOMKilled",
    text: `2026-10-29 11:22:04 [kubelet-worker-node-4] WARNING Pod eviction triggered for namespace prod-payments pod payment-processor-84f9bc7c4d-xk92m
[kubelet] Reason: EvictionThresholdMet - MemoryPressure condition active on node worker-4 (memory.available < 256Mi).
[containerd] Terminating container 'worker': OOMKilled exit_code=137 memory_limit=2048Mi memory_usage=2049Mi rss=1980Mi cache=69Mi
[INFO] Kafka consumer group 'payment-ingress-v2' rebalance initiated: consumer payment-processor-84f9bc7c4d-xk92m revoked 4 partitions [tx-part-0, tx-part-1, tx-part-2, tx-part-3].
[ERROR] Uncommitted transaction batch aborted for merchant_id=m_992144 amount_cents=450000 currency=USD.`,
    query: "Why was the payment processor pod evicted, what was the container exit code, and what happened to the Kafka consumer partitions?"
  },
  db: {
    title: "Database Pool Starvation",
    text: `2026-10-29 10:15:33 [HikariPool-1] ERROR com.zaxxer.hikari.pool.HikariPool - Connection leak detection triggered!
[HikariPool-1] Connection org.postgresql.jdbc.PgConnection@6b3e41a was borrowed at com.cloudcorp.billing.InvoiceReportGenerator.generateMonthlyLedger(InvoiceReportGenerator.java:156)
[WARN] Pool stats: (total=50, active=50, idle=0, waiting=142) - maximum connection pool size reached.
[ERROR] org.postgresql.util.PSQLException: Fatal timeout waiting for available database connection after 30000ms.
[INFO] Primary database instance db-pg-cluster-read-02 replica lag is 12ms, connection load 100%.`,
    query: "What caused the database failure, which method leaked the connection, and how many threads were waiting?"
  }
};

// WebGPU Supported Models Catalog
const WEBGPU_MODELS = [
  { id: "Qwen2.5-0.5B-Instruct-q4f16_1-MLC", label: "Qwen2.5-0.5B-Instruct (Fast ~350MB)" },
  { id: "SmolLM2-360M-Instruct-q4f16_1-MLC", label: "SmolLM2-360M-Instruct (Ultra-fast ~200MB)" },
  { id: "Llama-3.2-1B-Instruct-q4f16_1-MLC", label: "Llama-3.2-1B-Instruct (Meta ~800MB)" },
  { id: "Qwen2.5-1.5B-Instruct-q4f16_1-MLC", label: "Qwen2.5-1.5B-Instruct (~1GB)" }
];

// DOM References
const statTotal = document.getElementById("statTotal");
const statPruned = document.getElementById("statPruned");
const statSaved = document.getElementById("statSaved");
const statLatency = document.getElementById("statLatency");

const rawContext = document.getElementById("rawContext");
const diffOutput = document.getElementById("diffOutput");
const queryInput = document.getElementById("queryInput");
const threshRange = document.getElementById("threshRange");
const threshVal = document.getElementById("threshVal");
const runBtn = document.getElementById("runBtn");
const pruneOnlyBtn = document.getElementById("pruneOnlyBtn");
const modelSelect = document.getElementById("modelSelect");
const streamStatusBadge = document.getElementById("streamStatusBadge");
const llmResponse = document.getElementById("llmResponse");

// Phase 2 DOM References
const queryConditionToggle = document.getElementById("queryConditionToggle");
const queryWeightRange = document.getElementById("queryWeightRange");
const queryWeightVal = document.getElementById("queryWeightVal");
const queryWeightWrap = document.getElementById("queryWeightWrap");
const shaderLatencyBadge = document.getElementById("shaderLatencyBadge");
const shaderLatencyText = document.getElementById("shaderLatencyText");

// Engine Switcher References
const engineOllamaBtn = document.getElementById("engineOllamaBtn");
const engineWebGpuBtn = document.getElementById("engineWebGpuBtn");

const webgpuProgressCard = document.getElementById("webgpuProgressCard");
const webgpuProgressText = document.getElementById("webgpuProgressText");
const webgpuProgressPct = document.getElementById("webgpuProgressPct");
const webgpuProgressBar = document.getElementById("webgpuProgressBar");

const clearContextBtn = document.getElementById("clearContextBtn");
const pasteSampleBtn = document.getElementById("pasteSampleBtn");
const copyCompressedBtn = document.getElementById("copyCompressedBtn");
const copyResponseBtn = document.getElementById("copyResponseBtn");

const providerBadge = document.getElementById("providerBadge");
const providerText = document.getElementById("providerText");
const ollamaBadge = document.getElementById("ollamaBadge");
const ollamaText = document.getElementById("ollamaText");
const webgpuBadge = document.getElementById("webgpuBadge");
const webgpuText = document.getElementById("webgpuText");

const tokenTooltip = document.getElementById("tokenTooltip");
const tooltipWord = document.getElementById("tooltipWord");
const tooltipStatus = document.getElementById("tooltipStatus");
const tooltipScore = document.getElementById("tooltipScore");
const tooltipSpan = document.getElementById("tooltipSpan");

// State
let lastCompressedText = "";
let currentAbortController = null;
let currentPrunedData = null; // Raw backend result
let currentActiveTokens = []; // Tokens currently rendered (conditioned or base)
let currentEngine = "ollama"; // "ollama" or "webgpu"
let ollamaModelsCache = ["llama3.2:latest", "qwen2.5:7b"];

// WebLLM State
let webllmModule = null;
let webllmEngine = null;
let currentLoadedWebgpuModel = null;

// Phase 2 WebGPU Shader Instance
const queryPruner = new WebGPUQueryPruner(64);

// Initialize
document.addEventListener("DOMContentLoaded", () => {
  // Pre-fill default diagnostic log
  loadPreset("auth");

  // Setup Event Listeners
  threshRange.addEventListener("input", handleThresholdSliderChange);
  runBtn.addEventListener("click", handlePruneAndRun);
  pruneOnlyBtn.addEventListener("click", handlePruneOnly);

  // Phase 2 Event Listeners
  queryConditionToggle.addEventListener("change", handleQueryConditionToggle);
  queryWeightRange.addEventListener("input", handleQueryWeightChange);
  queryInput.addEventListener("input", debounce(handleQueryInputChange, 150));

  // Engine Switcher Listeners
  engineOllamaBtn.addEventListener("click", () => switchEngine("ollama"));
  engineWebGpuBtn.addEventListener("click", () => switchEngine("webgpu"));

  clearContextBtn.addEventListener("click", () => {
    rawContext.value = "";
    diffOutput.innerHTML = `<div class="diff-placeholder">Paste your logs and click <strong>"Prune & Run LLM"</strong></div>`;
    resetMetrics();
  });

  pasteSampleBtn.addEventListener("click", () => {
    loadPreset("auth");
  });

  copyCompressedBtn.addEventListener("click", () => {
    if (lastCompressedText) {
      navigator.clipboard.writeText(lastCompressedText);
      showTemporaryButtonState(copyCompressedBtn, "Copied!");
    } else {
      showTemporaryButtonState(copyCompressedBtn, "Nothing to copy");
    }
  });

  copyResponseBtn.addEventListener("click", () => {
    const text = llmResponse.innerText.trim();
    if (text) {
      navigator.clipboard.writeText(text);
      showTemporaryButtonState(copyResponseBtn, "Copied!");
    }
  });

  // Preset pill click handlers
  document.querySelectorAll(".preset-pill").forEach(pill => {
    pill.addEventListener("click", () => {
      document.querySelectorAll(".preset-pill").forEach(p => p.classList.remove("active"));
      pill.classList.add("active");
      const key = pill.getAttribute("data-preset");
      loadPreset(key);
    });
  });

  // Pre-initialize WebGPU Shader engine and check hardware
  initWebGpuShader();

  // Check health and models
  checkBackendHealth();
});

// Pre-initialize WebGPU Shader
async function initWebGpuShader() {
  const supported = await queryPruner.init();
  if (supported) {
    webgpuBadge.className = "status-badge webgpu-ready";
    webgpuText.innerText = "WebGPU Ready";
    shaderLatencyBadge.className = "shader-badge";
    shaderLatencyText.innerText = "WebGPU Shader: Ready";
  } else {
    webgpuBadge.className = "status-badge";
    webgpuText.innerText = "WebGPU (CPU Fallback)";
    shaderLatencyBadge.className = "shader-badge idle";
    shaderLatencyText.innerText = "WebGPU Shader: CPU Mode";
  }
}

// Handle Query Conditioning Toggle
async function handleQueryConditionToggle() {
  const isEnabled = queryConditionToggle.checked;
  queryWeightWrap.style.opacity = isEnabled ? "1" : "0.4";
  queryWeightWrap.style.pointerEvents = isEnabled ? "auto" : "none";

  if (!currentPrunedData) return;

  if (isEnabled) {
    await applyQueryConditioningAndRender();
  } else {
    shaderLatencyBadge.className = "shader-badge idle";
    shaderLatencyText.innerText = "WebGPU Shader: Disabled";
    restoreBaselinePruning();
  }
}

// Handle Query Weight Range Slider
async function handleQueryWeightChange() {
  const weight = parseFloat(queryWeightRange.value).toFixed(2);
  queryWeightVal.innerText = weight;

  if (queryConditionToggle.checked && currentPrunedData) {
    await applyQueryConditioningAndRender();
  }
}

// Handle real-time typing in Query Input box
async function handleQueryInputChange() {
  if (queryConditionToggle.checked && currentPrunedData) {
    await applyQueryConditioningAndRender();
  }
}

/**
 * Executes the WebGPU WGSL Cross-Attention Shader:
 * Takes context tokens, base LLMLingua-2 scores, and the query,
 * and updates token retention probabilities in real time (< 0.5 ms).
 */
async function applyQueryConditioningAndRender() {
  if (!currentPrunedData || !currentPrunedData.tokens || currentPrunedData.tokens.length === 0) return;

  const query = queryInput.value.trim();
  const threshold = parseFloat(threshRange.value);
  const queryWeight = parseFloat(queryWeightRange.value); // (1 - alpha)
  const alpha = 1.0 - queryWeight; // Weight for base LLMLingua score

  const baseTokens = currentPrunedData.tokens;
  const tokenWords = baseTokens.map(t => t.word);
  const baseScores = baseTokens.map(t => t.score);

  // Run WebGPU Compute Shader
  shaderLatencyBadge.className = "shader-badge";
  const { scores, attention, gpuLatencyMs } = await queryPruner.conditionScores(
    tokenWords,
    baseScores,
    query,
    alpha
  );

  shaderLatencyText.innerText = `WebGPU Shader: ${gpuLatencyMs} ms`;

  // Build conditioned token list
  let prunedCount = 0;
  const conditionedTokens = baseTokens.map((t, i) => {
    const finalScore = parseFloat(scores[i].toFixed(4));
    const attnScore = parseFloat(attention[i].toFixed(4));
    const isKept = finalScore >= threshold;
    if (!isKept) prunedCount++;

    return {
      ...t,
      score: finalScore,
      base_score: t.score,
      query_attn: attnScore,
      is_kept: isKept,
    };
  });

  currentActiveTokens = conditionedTokens;

  // Recalculate metrics
  const total = conditionedTokens.length;
  const savedPct = total > 0 ? (prunedCount / total * 100) : 0;

  statPruned.innerText = prunedCount;
  statSaved.innerText = savedPct.toFixed(1);

  // Reconstruct compressed text
  reconstructCompressedText(rawContext.value, conditionedTokens);

  // Render visual diff with conditioned scores
  renderTokenDiff(rawContext.value, conditionedTokens);
}

// Restore baseline task-agnostic LLMLingua-2 pruning
function restoreBaselinePruning() {
  if (!currentPrunedData) return;
  const threshold = parseFloat(threshRange.value);

  let prunedCount = 0;
  const baseTokens = currentPrunedData.tokens.map(t => {
    const isKept = t.score >= threshold;
    if (!isKept) prunedCount++;
    return {
      ...t,
      is_kept: isKept,
      query_attn: null,
      base_score: t.score,
    };
  });

  currentActiveTokens = baseTokens;
  const total = baseTokens.length;
  const savedPct = total > 0 ? (prunedCount / total * 100) : 0;

  statPruned.innerText = prunedCount;
  statSaved.innerText = savedPct.toFixed(1);

  reconstructCompressedText(rawContext.value, baseTokens);
  renderTokenDiff(rawContext.value, baseTokens);
}

// Reconstruct compressed text from kept tokens preserving spacing
function reconstructCompressedText(originalText, tokens) {
  const parts = [];
  let lastEnd = -1;

  for (const t of tokens) {
    if (t.is_kept) {
      if (lastEnd !== -1) {
        const between = originalText.substring(lastEnd, t.start);
        if (between.includes("\n")) {
          parts.push("\n");
        } else if (between.trim() === "" && between.length > 0) {
          parts.push(" ");
        } else if (t.start > lastEnd) {
          parts.push(" ");
        }
      }
      parts.push(originalText.substring(t.start, t.end));
      lastEnd = t.end;
    }
  }

  lastCompressedText = parts.join("").trim();
}

// Switch generation runtime engine
function switchEngine(engine) {
  currentEngine = engine;
  if (engine === "ollama") {
    engineOllamaBtn.classList.add("active");
    engineWebGpuBtn.classList.remove("active");
    populateOllamaModels();
  } else {
    engineWebGpuBtn.classList.add("active");
    engineOllamaBtn.classList.remove("active");
    populateWebGpuModels();
  }
}

function populateOllamaModels() {
  modelSelect.innerHTML = "";
  ollamaModelsCache.forEach(model => {
    const opt = document.createElement("option");
    opt.value = model;
    opt.innerText = model;
    modelSelect.appendChild(opt);
  });
}

function populateWebGpuModels() {
  modelSelect.innerHTML = "";
  WEBGPU_MODELS.forEach(m => {
    const opt = document.createElement("option");
    opt.value = m.id;
    opt.innerText = m.label;
    modelSelect.appendChild(opt);
  });
}

// Dynamically load WebLLM module only when needed
async function getWebLLM() {
  if (!webllmModule) {
    webgpuProgressCard.style.display = "flex";
    webgpuProgressText.innerText = "Importing WebLLM engine module...";
    webgpuProgressPct.innerText = "10%";
    webgpuProgressBar.style.width = "10%";
    webllmModule = await import("https://esm.run/@mlc-ai/web-llm");
  }
  return webllmModule;
}

// Ensure WebGPU model is loaded into browser cache & VRAM
async function ensureWebGpuEngine(modelId) {
  if (webllmEngine && currentLoadedWebgpuModel === modelId) {
    return webllmEngine;
  }

  const lib = await getWebLLM();
  webgpuProgressCard.style.display = "flex";

  const initProgressCallback = (report) => {
    webgpuProgressText.innerText = report.text || "Loading model weights...";
    const pct = Math.round((report.progress || 0) * 100);
    webgpuProgressPct.innerText = `${pct}%`;
    webgpuProgressBar.style.width = `${pct}%`;
  };

  webllmEngine = await lib.CreateMLCEngine(modelId, { initProgressCallback });
  currentLoadedWebgpuModel = modelId;
  webgpuProgressCard.style.display = "none";
  return webllmEngine;
}

// Load preset logs
function loadPreset(key) {
  const preset = PRESET_LOGS[key] || PRESET_LOGS.auth;
  rawContext.value = preset.text;
  queryInput.value = preset.query;
}

// Threshold Slider change
async function handleThresholdSliderChange() {
  const val = parseFloat(threshRange.value).toFixed(2);
  threshVal.innerText = val;

  if (queryConditionToggle.checked && currentPrunedData) {
    await applyQueryConditioningAndRender();
  } else if (currentPrunedData) {
    restoreBaselinePruning();
  }
}

// Check Backend Health and Populate Models
async function checkBackendHealth() {
  try {
    const res = await fetch(`${API_BASE}/api/health`);
    if (res.ok) {
      const data = await res.json();
      if (data.provider) {
        if (data.provider.includes("CUDA")) {
          providerBadge.className = "status-badge gpu";
          providerText.innerText = "CUDA RTX 5060 Accelerated";
        } else {
          providerBadge.className = "status-badge";
          providerText.innerText = "CPU Execution Provider";
        }
      }

      if (data.ollama_online) {
        ollamaBadge.className = "status-badge active";
        ollamaText.innerText = `Ollama Online (${data.available_models[0] || 'active'})`;
      } else {
        ollamaBadge.className = "status-badge";
        ollamaText.innerText = "Ollama Offline";
      }

      // Cache Ollama models
      if (data.available_models && data.available_models.length > 0) {
        ollamaModelsCache = [...data.available_models];
        if (!ollamaModelsCache.includes("qwen2.5:7b")) {
          ollamaModelsCache.push("qwen2.5:7b");
        }
      }
      if (currentEngine === "ollama") {
        populateOllamaModels();
      }
    }
  } catch (err) {
    console.warn("Backend health check failed:", err);
  }
}

// Reset HUD metrics
function resetMetrics() {
  statTotal.innerText = "0";
  statPruned.innerText = "0";
  statSaved.innerText = "0.0";
  statLatency.innerText = "0.0";
  shaderLatencyBadge.className = "shader-badge idle";
  shaderLatencyText.innerText = "WebGPU Shader: Ready";
}

// Animate numbers for HUD cards
function animateValue(elem, start, end, duration = 300, isFloat = false) {
  const startTime = performance.now();
  function update(currentTime) {
    const elapsed = currentTime - startTime;
    const progress = Math.min(elapsed / duration, 1);
    const current = start + (end - start) * progress;
    elem.innerText = isFloat ? current.toFixed(1) : Math.round(current);
    if (progress < 1) {
      requestAnimationFrame(update);
    } else {
      elem.innerText = isFloat ? end.toFixed(1) : Math.round(end);
    }
  }
  requestAnimationFrame(update);
}

// Prune text via backend API
async function executePrune(text, threshold) {
  const response = await fetch(`${API_BASE}/api/prune`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, threshold })
  });

  if (!response.ok) {
    throw new Error(`Prune API error: ${response.status} ${await response.text()}`);
  }

  return await response.json();
}

// Render Token Diff in Right Pane
function renderTokenDiff(originalText, tokens) {
  diffOutput.innerHTML = "";

  if (!tokens || tokens.length === 0) {
    diffOutput.innerHTML = `<div class="diff-placeholder">No tokens found.</div>`;
    return;
  }

  const fragment = document.createDocumentFragment();
  let lastCharIndex = 0;

  tokens.forEach(tok => {
    // Preserve any whitespace/newlines between previous token and current token
    if (tok.start > lastCharIndex) {
      const interstitial = originalText.substring(lastCharIndex, tok.start);
      fragment.appendChild(document.createTextNode(interstitial));
    }

    const span = document.createElement("span");
    span.className = tok.is_kept ? "token-kept" : "token-pruned";
    span.textContent = tok.word;

    // Attach data attributes for tooltip and local re-calculation
    span.dataset.word = tok.word;
    span.dataset.score = tok.score;
    span.dataset.baseScore = tok.base_score || tok.score;
    span.dataset.queryAttn = tok.query_attn !== undefined && tok.query_attn !== null ? tok.query_attn : "";
    span.dataset.kept = tok.is_kept;
    span.dataset.start = tok.start;
    span.dataset.end = tok.end;

    // Tooltip event handlers
    span.addEventListener("mouseenter", showTooltip);
    span.addEventListener("mousemove", moveTooltip);
    span.addEventListener("mouseleave", hideTooltip);

    fragment.appendChild(span);
    lastCharIndex = tok.end;
  });

  // Append remaining text after last token if any
  if (lastCharIndex < originalText.length) {
    fragment.appendChild(document.createTextNode(originalText.substring(lastCharIndex)));
  }

  diffOutput.appendChild(fragment);
}

// Tooltip positioning and content
function showTooltip(e) {
  const target = e.currentTarget;
  const word = target.dataset.word || target.textContent;
  const score = parseFloat(target.dataset.score || 0);
  const baseScore = parseFloat(target.dataset.baseScore || score);
  const queryAttn = target.dataset.queryAttn ? parseFloat(target.dataset.queryAttn) : null;
  const isKept = target.dataset.kept === "true";
  const start = target.dataset.start || 0;
  const end = target.dataset.end || 0;

  tooltipWord.textContent = `"${word}"`;
  tooltipStatus.textContent = isKept ? "RETAINED" : "EVICTED";
  tooltipStatus.style.color = isKept ? "var(--accent-emerald)" : "var(--accent-rose)";

  if (queryAttn !== null && queryConditionToggle.checked) {
    tooltipScore.innerHTML = `
      <div>Final: <strong>${(score * 100).toFixed(1)}%</strong> (p=${score.toFixed(3)})</div>
      <div style="font-size:0.7rem; color:var(--text-muted); margin-top:2px;">
        Base Lingua: ${(baseScore * 100).toFixed(0)}% &bull; Query Attn: ${(queryAttn * 100).toFixed(0)}%
      </div>
    `;
  } else {
    tooltipScore.textContent = `${(score * 100).toFixed(1)}% (p=${score.toFixed(4)})`;
  }

  tooltipSpan.textContent = `offset [${start}..${end}]`;

  moveTooltip(e);
  tokenTooltip.style.display = "block";
}

function moveTooltip(e) {
  const x = e.clientX + 14;
  const y = e.clientY + 14;
  tokenTooltip.style.left = `${Math.min(x, window.innerWidth - 300)}px`;
  tokenTooltip.style.top = `${Math.min(y, window.innerHeight - 130)}px`;
}

function hideTooltip() {
  tokenTooltip.style.display = "none";
}

// Handle "Prune Only" button
async function handlePruneOnly() {
  const text = rawContext.value.trim();
  if (!text) {
    alert("Please provide input text or diagnostic logs in the left pane.");
    return;
  }

  const threshold = parseFloat(threshRange.value);
  setLoadingState(true, "Pruning...");

  try {
    const data = await executePrune(text, threshold);
    currentPrunedData = data;

    // Update HUD metrics with smooth transitions
    animateValue(statTotal, parseInt(statTotal.innerText) || 0, data.total_tokens);
    statLatency.innerText = data.latency_ms.toFixed(1);

    // If Phase 2 Query-Conditioning is active, run WebGPU shader
    if (queryConditionToggle.checked) {
      await applyQueryConditioningAndRender();
    } else {
      currentActiveTokens = data.tokens;
      animateValue(statPruned, parseInt(statPruned.innerText) || 0, data.pruned_tokens);
      animateValue(statSaved, parseFloat(statSaved.innerText) || 0, data.saved_pct, 300, true);
      lastCompressedText = data.compressed_text;
      renderTokenDiff(text, data.tokens);
    }
  } catch (err) {
    console.error("Pruning error:", err);
    alert(`Failed to prune text: ${err.message}`);
  } finally {
    setLoadingState(false);
  }
}

// Handle "Prune & Run LLM" button (Ollama SSE or WebGPU in-browser)
async function handlePruneAndRun() {
  const text = rawContext.value.trim();
  const query = queryInput.value.trim();

  if (!text) {
    alert("Please provide input text or diagnostic logs in the left pane.");
    return;
  }
  if (!query) {
    alert("Please specify a query for the LLM.");
    return;
  }

  // Abort any ongoing stream
  if (currentAbortController) {
    currentAbortController.abort();
  }
  currentAbortController = new AbortController();

  const threshold = parseFloat(threshRange.value);
  const selectedModel = modelSelect.value || (currentEngine === "ollama" ? "llama3.2:latest" : WEBGPU_MODELS[0].id);

  setLoadingState(true, "Pruning...");

  try {
    // Step 1: Prune tokens on GPU via backend
    const data = await executePrune(text, threshold);
    currentPrunedData = data;

    // Update HUD Metrics
    animateValue(statTotal, parseInt(statTotal.innerText) || 0, data.total_tokens);
    statLatency.innerText = data.latency_ms.toFixed(1);

    // Apply Phase 2 Query-Conditioned WebGPU Shader if enabled
    if (queryConditionToggle.checked) {
      await applyQueryConditioningAndRender();
    } else {
      currentActiveTokens = data.tokens;
      animateValue(statPruned, parseInt(statPruned.innerText) || 0, data.pruned_tokens);
      animateValue(statSaved, parseFloat(statSaved.innerText) || 0, data.saved_pct, 300, true);
      lastCompressedText = data.compressed_text;
      renderTokenDiff(text, data.tokens);
    }

    // Reset response view
    llmResponse.innerHTML = `<span id="streamContent"></span><span class="cursor-blink"></span>`;
    const streamContent = document.getElementById("streamContent");

    // Step 2: Stream LLM Generation
    if (currentEngine === "webgpu") {
      // IN-BROWSER WEBGPU EXECUTION
      setLoadingState(true, "Loading WebGPU...");
      streamStatusBadge.className = "stream-status-badge streaming";
      streamStatusBadge.innerText = `WebGPU In-Browser (${selectedModel.split("-")[0]})`;

      const engine = await ensureWebGpuEngine(selectedModel);

      setLoadingState(true, "Streaming WebGPU...");
      const messages = [
        {
          role: "system",
          content: "You are an expert diagnostic assistant. Answer the user query factually based on the provided context."
        },
        {
          role: "user",
          content: `Context:\n${lastCompressedText}\n\nQuestion:\n${query}\n\nAnswer:`
        }
      ];

      const asyncChunkGenerator = await engine.chat.completions.create({
        messages,
        stream: true,
      });

      for await (const chunk of asyncChunkGenerator) {
        if (currentAbortController.signal.aborted) break;
        const delta = chunk.choices[0]?.delta?.content || "";
        if (delta) {
          streamContent.textContent += delta;
          llmResponse.scrollTop = llmResponse.scrollHeight;
        }
      }

    } else {
      // OLLAMA SSE STREAMING EXECUTION
      setLoadingState(true, "Streaming Ollama...");
      streamStatusBadge.className = "stream-status-badge streaming";
      streamStatusBadge.innerText = `Streaming Ollama (${selectedModel})`;

      const genResponse = await fetch(`${API_BASE}/api/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          compressed_context: lastCompressedText,
          query: query,
          model: selectedModel
        }),
        signal: currentAbortController.signal
      });

      if (!genResponse.ok) {
        throw new Error(`Generation endpoint error: ${genResponse.status}`);
      }

      // Consume SSE stream using ReadableStream and TextDecoder
      const reader = genResponse.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let buffer = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n\n");
        buffer = lines.pop(); // Keep incomplete chunk in buffer

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed.startsWith("data:")) continue;

          const payloadStr = trimmed.replace(/^data:\s*/, "");
          if (payloadStr === "[DONE]") {
            break;
          }

          try {
            const parsed = jsonParseSafe(payloadStr);
            if (parsed && parsed.token) {
              streamContent.textContent += parsed.token;
              llmResponse.scrollTop = llmResponse.scrollHeight;
            }
          } catch (e) {
            if (payloadStr) {
              streamContent.textContent += payloadStr;
            }
          }
        }
      }
    }

    // Generation Complete
    streamStatusBadge.className = "stream-status-badge";
    streamStatusBadge.innerText = "Completed";
    const cursor = llmResponse.querySelector(".cursor-blink");
    if (cursor) cursor.remove();

  } catch (err) {
    if (err.name === "AbortError") {
      streamStatusBadge.innerText = "Cancelled";
    } else {
      console.error("Pipeline error:", err);
      llmResponse.innerHTML = `<div style="color:var(--accent-rose); padding:1rem;">Error: ${err.message}</div>`;
      streamStatusBadge.innerText = "Failed";
    }
  } finally {
    setLoadingState(false);
  }
}

// Helpers
function jsonParseSafe(str) {
  try {
    return JSON.parse(str);
  } catch {
    return null;
  }
}

function debounce(fn, ms = 200) {
  let timer = null;
  return function (...args) {
    clearTimeout(timer);
    timer = setTimeout(() => fn.apply(this, args), ms);
  };
}

function setLoadingState(isLoading, label = "") {
  runBtn.disabled = isLoading;
  pruneOnlyBtn.disabled = isLoading;

  if (isLoading) {
    runBtn.innerHTML = `<span class="spinner"></span> ${label}`;
  } else {
    runBtn.innerHTML = `
      <svg width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M13 10V3L4 14h7v7l9-11h-7z"/></svg>
      Prune &amp; Run LLM
    `;
    pruneOnlyBtn.innerHTML = `
      <svg width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M14.121 14.121L19 19m-7-7l7-7m-7 7l-2.879 2.879a3 3 0 11-4.242-4.242L10.757 5.5M7 17l-4 4"/></svg>
      Prune Only
    `;
  }
}

function showTemporaryButtonState(btn, text) {
  const orig = btn.innerText;
  btn.innerText = text;
  setTimeout(() => {
    btn.innerText = orig;
  }, 1800);
}
