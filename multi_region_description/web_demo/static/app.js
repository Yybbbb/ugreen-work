// ─── DOM refs ────────────────────────────────────────────────────────────────
const imageInput        = document.getElementById("imageInput");
const imageCanvas       = document.getElementById("imageCanvas");
const ctx               = imageCanvas.getContext("2d");
const dropZone          = document.getElementById("dropZone");
const predictButton     = document.getElementById("predictButton");
const clearBoxesButton  = document.getElementById("clearBoxesButton");
const bboxText          = document.getElementById("bboxText");
const bboxMode          = document.getElementById("bboxMode");
const loadTextBoxesButton = document.getElementById("loadTextBoxesButton");
const copyBoxesButton   = document.getElementById("copyBoxesButton");
const boxList           = document.getElementById("boxList");
const boxCount          = document.getElementById("boxCount");
const resultsBody       = document.getElementById("resultsBody");
const resultMeta        = document.getElementById("resultMeta");
const statusText        = document.getElementById("statusText");
const imageMeta         = document.getElementById("imageMeta");
const rawPanel          = document.getElementById("rawPanel");
const rawOutput         = document.getElementById("rawOutput");
const rawMeta           = document.getElementById("rawMeta");
const modelSelect       = document.getElementById("modelSelect");

// ─── State ───────────────────────────────────────────────────────────────────
let imageFile    = null;
let imageBitmap  = null;
let imageWidth   = 0;
let imageHeight  = 0;
let boxes        = [];       // [{x1,y1,x2,y2}]  pixel coords
let drawing      = null;     // in-progress drag: {start, currentBox}
let lastResults  = [];       // last /api/predict results array

// ─── Status helpers ───────────────────────────────────────────────────────────
function setStatus(text, isError = false) {
  statusText.textContent = text;
  statusText.className = isError ? "errorText" : "";
}
function setResultMeta(text, isError = false) {
  resultMeta.textContent = text;
  resultMeta.className = isError ? "errorText" : "";
}

// ─── Canvas sizing ───────────────────────────────────────────────────────────
function fitCanvas() {
  const wrap = imageCanvas.parentElement;
  if (!imageBitmap || !wrap) { imageCanvas.width = 960; imageCanvas.height = 540; return; }
  const maxW = Math.max(320, wrap.clientWidth - 24);
  const maxH = Math.max(260, wrap.clientHeight - 24);
  const scale = Math.min(maxW / imageWidth, maxH / imageHeight, 1);
  imageCanvas.width  = Math.max(1, Math.round(imageWidth  * scale));
  imageCanvas.height = Math.max(1, Math.round(imageHeight * scale));
}

// ─── Coordinate helpers ───────────────────────────────────────────────────────
function canvasToImagePoint(event) {
  const rect = imageCanvas.getBoundingClientRect();
  return {
    x: Math.max(0, Math.min(imageWidth,  ((event.clientX - rect.left) / rect.width)  * imageWidth)),
    y: Math.max(0, Math.min(imageHeight, ((event.clientY - rect.top)  / rect.height) * imageHeight)),
  };
}
function imageToCanvasBox(box) {
  const sx = imageCanvas.width / imageWidth, sy = imageCanvas.height / imageHeight;
  return { x: box.x1 * sx, y: box.y1 * sy, w: (box.x2 - box.x1) * sx, h: (box.y2 - box.y1) * sy };
}
function pixelToLoc(box) {
  const bw = imageWidth / 1000, bh = imageHeight / 1000;
  return [
    Math.max(0, Math.min(999, Math.floor(box.x1 / bw))),
    Math.max(0, Math.min(999, Math.floor(box.y1 / bh))),
    Math.max(0, Math.min(999, Math.floor(box.x2 / bw))),
    Math.max(0, Math.min(999, Math.floor(box.y2 / bh))),
  ];
}
function locToPixel(loc) {
  return { x1: loc[0]*imageWidth/1000, y1: loc[1]*imageHeight/1000,
           x2: loc[2]*imageWidth/1000, y2: loc[3]*imageHeight/1000 };
}
function normalizeBox(a, b) {
  return { x1: Math.min(a.x, b.x), y1: Math.min(a.y, b.y),
           x2: Math.max(a.x, b.x), y2: Math.max(a.y, b.y) };
}
function isValidBox(box) { return box.x2 - box.x1 >= 3 && box.y2 - box.y1 >= 3; }
function formatBox(box) { return [box.x1, box.y1, box.x2, box.y2].map(v => Math.round(v*10)/10).join(","); }

// ─── Box color palette (up to 12 distinct colors) ────────────────────────────
const BOX_COLORS = [
  "#0f766e","#1d4ed8","#b45309","#7c3aed","#0369a1",
  "#be123c","#065f46","#92400e","#1e3a5f","#4a044e",
  "#14532d","#7f1d1d",
];
function boxColor(index) { return BOX_COLORS[index % BOX_COLORS.length]; }

// ─── Draw ─────────────────────────────────────────────────────────────────────
function drawBox(box, index, color, label = null) {
  const cb = imageToCanvasBox(box);
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  const fillAlpha = label ? "29" : "24";   // slightly stronger fill when results shown
  ctx.fillStyle = color + fillAlpha;
  ctx.fillRect(cb.x, cb.y, cb.w, cb.h);
  ctx.strokeRect(cb.x, cb.y, cb.w, cb.h);

  // index badge
  const BADGE_W = 30, BADGE_H = 20;
  const bx = cb.x, by = Math.max(0, cb.y - BADGE_H);
  ctx.fillStyle = color;
  ctx.fillRect(bx, by, BADGE_W, BADGE_H);
  ctx.fillStyle = "white";
  ctx.font = "bold 12px ui-monospace, monospace";
  ctx.fillText(String(index), bx + 7, by + 14);

  // description label (truncated) below the badge, inside or below the box
  if (label) {
    const MAX_CHARS = 52;
    const text = label.length > MAX_CHARS ? label.slice(0, MAX_CHARS - 1) + "…" : label;
    ctx.font = "12px ui-sans-serif, sans-serif";
    const textW = ctx.measureText(text).width + 10;
    const ly = Math.min(cb.y + cb.h - 6, cb.y + cb.h + 18);
    const lx = cb.x;
    ctx.fillStyle = color + "cc";
    ctx.fillRect(lx, ly - 16, textW, 20);
    ctx.fillStyle = "white";
    ctx.fillText(text, lx + 5, ly - 2);
  }
  ctx.restore();
}

function redraw() {
  ctx.clearRect(0, 0, imageCanvas.width, imageCanvas.height);
  if (imageBitmap) {
    ctx.drawImage(imageBitmap, 0, 0, imageCanvas.width, imageCanvas.height);
  } else {
    ctx.fillStyle = "#17211f";
    ctx.fillRect(0, 0, imageCanvas.width, imageCanvas.height);
    ctx.fillStyle = "#dce5df";
    ctx.font = "18px ui-sans-serif, sans-serif";
    ctx.fillText("Open an image to begin", 28, 46);
  }
  boxes.forEach((box, i) => {
    const result = lastResults[i];
    const label = result?.description ?? null;
    drawBox(box, i + 1, boxColor(i), label);
  });
  if (drawing?.currentBox && isValidBox(drawing.currentBox)) {
    drawBox(drawing.currentBox, boxes.length + 1, "#b45309");
  }
}

// ─── Box list sidebar ─────────────────────────────────────────────────────────
function syncBoxList() {
  boxCount.textContent = String(boxes.length);
  predictButton.disabled = !imageFile || boxes.length === 0;
  if (!boxes.length) {
    boxList.innerHTML = '<div class="empty">No boxes</div>';
    return;
  }
  boxList.innerHTML = "";
  boxes.forEach((box, i) => {
    const loc = imageBitmap ? pixelToLoc(box) : [];
    const item = document.createElement("div");
    item.className = "boxItem";
    item.innerHTML = `
      <div class="boxItemHeader">
        <strong style="color:${boxColor(i)}">#${i + 1}</strong>
        <button class="deleteBox" type="button" data-index="${i}">Delete</button>
      </div>
      <code>${formatBox(box)}</code>
      <code>loc: ${loc.join(",")}</code>
    `;
    boxList.appendChild(item);
  });
  boxList.querySelectorAll(".deleteBox").forEach(btn => {
    btn.addEventListener("click", () => {
      boxes.splice(Number(btn.dataset.index), 1);
      lastResults = [];
      redraw();
      syncBoxList();
      clearResults();
    });
  });
}

// ─── Manual bbox text parsing ─────────────────────────────────────────────────
function parseManualLines(text) {
  return text.split(/\n+/).map(l => l.trim()).filter(Boolean).map((line, i) => {
    const vals = line.replaceAll(",", " ").split(/\s+/).filter(Boolean).map(Number);
    if (vals.length !== 4 || vals.some(v => !Number.isFinite(v)))
      throw new Error(`line ${i + 1}: need four numeric values`);
    if (bboxMode.value === "loc") {
      if (vals.some(v => v < 0 || v > 999)) throw new Error(`line ${i + 1}: loc values must be in [0,999]`);
      return locToPixel(vals.map(v => Math.round(v)));
    }
    return { x1: vals[0], y1: vals[1], x2: vals[2], y2: vals[3] };
  });
}
function validateBoxes(bs) {
  bs.forEach((b, i) => {
    if (!isValidBox(b)) throw new Error(`box ${i + 1}: too small`);
    if (b.x1 < 0 || b.y1 < 0 || b.x2 > imageWidth || b.y2 > imageHeight)
      throw new Error(`box ${i + 1}: outside image`);
  });
}

// ─── Results rendering ────────────────────────────────────────────────────────
function clearResults() {
  rawPanel.style.display = "none";
  rawOutput.textContent = "";
  rawMeta.textContent = "";
  resultsBody.innerHTML = "";
  setResultMeta("idle");
}

function renderResults(data) {
  // ── raw output block ──
  rawPanel.style.display = "";
  rawOutput.textContent = data.raw || "";
  const countOk = data.count_match;
  rawMeta.innerHTML = countOk
    ? `${data.descriptions.length} descriptions matched ${data.count} boxes ✓`
    : `<span class="mismatchBadge">⚠ count mismatch: ${data.descriptions.length} descriptions for ${data.count} boxes</span>`;

  // ── per-region table ──
  resultsBody.innerHTML = "";
  if (!data.results?.length) {
    resultsBody.innerHTML = '<tr><td colspan="4" class="empty">No results</td></tr>';
    return;
  }
  for (const row of data.results) {
    const tr = document.createElement("tr");
    // # with color dot
    const tdIdx = document.createElement("td");
    tdIdx.innerHTML = `<span style="color:${boxColor(row.index)};font-weight:700">#${row.index + 1}</span>`;
    tr.appendChild(tdIdx);
    // pixel bbox
    const tdPixel = document.createElement("td");
    const codePixel = document.createElement("code");
    codePixel.textContent = (row.bbox_pixel || []).join(",");
    tdPixel.appendChild(codePixel);
    tr.appendChild(tdPixel);
    // loc bbox
    const tdLoc = document.createElement("td");
    const codeLoc = document.createElement("code");
    codeLoc.textContent = (row.bbox_loc_0_999 || []).join(",");
    tdLoc.appendChild(codeLoc);
    tr.appendChild(tdLoc);
    // description
    const tdDesc = document.createElement("td");
    tdDesc.className = "descriptionCell";
    if (row.description != null) {
      tdDesc.textContent = row.description;
    } else {
      tdDesc.innerHTML = '<span class="missingCell">— (count mismatch, no description)</span>';
    }
    tr.appendChild(tdDesc);
    resultsBody.appendChild(tr);
  }
}

// ─── API calls ────────────────────────────────────────────────────────────────
async function loadServiceStatus() {
  try {
    const res = await fetch("/api/health");
    const data = await res.json();
    const ma = data.models?.plan_a;
    const mb = data.models?.plan_b;
    setStatus(
      `${data.precision} · max_tokens ${data.default_max_new_tokens} · ` +
      `Plan A ${ma?.loaded ? "✓" : "idle"} · Plan B ${mb?.loaded ? "✓" : "idle"}`
    );
  } catch (err) {
    setStatus(`service unavailable: ${err.message}`, true);
  }
}

async function runPredict() {
  if (!imageFile) throw new Error("open image first");
  if (!boxes.length) throw new Error("add at least one bbox");

  const mode = modelSelect.value;
  setResultMeta(`running inference [${mode}]…`);
  predictButton.disabled = true;
  lastResults = [];
  clearResults();

  const form = new FormData();
  form.append("image", imageFile);
  form.append("boxes_json", JSON.stringify(
    boxes.map(b => ({ mode: "pixel", bbox: [b.x1, b.y1, b.x2, b.y2] }))
  ));
  form.append("mode", mode);

  const res = await fetch("/api/predict", { method: "POST", body: form });
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || `request failed: ${res.status}`);

  lastResults = data.results || [];
  renderResults(data);
  redraw();  // re-draw canvas with description labels on boxes
  setResultMeta(
    data.count_match
      ? `done [${mode}] · ${data.count} boxes · 1 inference call ✓`
      : `done [${mode}] · count mismatch ⚠ (${data.descriptions.length}/${data.count})`
  );
  await loadServiceStatus();
}

// ─── Image loading ────────────────────────────────────────────────────────────
async function loadImageFile(file) {
  if (!file) return;
  if (!file.type.startsWith("image/")) throw new Error("drop or open an image file");
  imageFile = file;
  imageBitmap = await createImageBitmap(file);
  imageWidth = imageBitmap.width;
  imageHeight = imageBitmap.height;
  boxes = [];
  lastResults = [];
  fitCanvas();
  redraw();
  syncBoxList();
  clearResults();
  imageMeta.textContent = `${file.name} · ${imageWidth}×${imageHeight}`;
}

// ─── Event listeners ──────────────────────────────────────────────────────────
imageInput.addEventListener("change", async () => {
  try { await loadImageFile(imageInput.files?.[0]); }
  catch (err) { setResultMeta(err.message, true); }
});

// drag-and-drop onto the canvas area
function markDragActive(e) { e.preventDefault(); e.stopPropagation(); dropZone.classList.add("dragActive"); }
dropZone.addEventListener("dragenter", markDragActive);
dropZone.addEventListener("dragover",  markDragActive);
dropZone.addEventListener("dragleave", (e) => {
  e.preventDefault(); e.stopPropagation();
  if (!dropZone.contains(e.relatedTarget)) dropZone.classList.remove("dragActive");
});
dropZone.addEventListener("drop", async (e) => {
  e.preventDefault(); e.stopPropagation();
  dropZone.classList.remove("dragActive");
  try { await loadImageFile(e.dataTransfer?.files?.[0]); }
  catch (err) { setResultMeta(err.message, true); }
});

// canvas drawing
imageCanvas.addEventListener("mousedown", (e) => {
  if (!imageBitmap) return;
  drawing = { start: canvasToImagePoint(e), currentBox: null };
});
imageCanvas.addEventListener("mousemove", (e) => {
  if (!drawing) return;
  drawing.currentBox = normalizeBox(drawing.start, canvasToImagePoint(e));
  redraw();
});
window.addEventListener("mouseup", () => {
  if (!drawing) return;
  if (drawing.currentBox && isValidBox(drawing.currentBox)) {
    boxes.push(drawing.currentBox);
    syncBoxList();
  }
  drawing = null;
  redraw();
});

clearBoxesButton.addEventListener("click", () => {
  boxes = []; lastResults = [];
  redraw(); syncBoxList(); clearResults();
});

loadTextBoxesButton.addEventListener("click", () => {
  try {
    if (!imageBitmap) throw new Error("open image first");
    const next = parseManualLines(bboxText.value);
    validateBoxes(next);
    boxes = next; lastResults = [];
    redraw(); syncBoxList();
    setResultMeta(`loaded ${boxes.length} boxes`);
  } catch (err) { setResultMeta(err.message, true); }
});

copyBoxesButton.addEventListener("click", async () => {
  const lines = boxes.map(b => formatBox(b)).join("\n");
  bboxText.value = lines;
  await navigator.clipboard?.writeText(lines).catch(() => undefined);
});

predictButton.addEventListener("click", async () => {
  try {
    await runPredict();
  } catch (err) {
    setResultMeta(err.message, true);
    lastResults = [];
    redraw();
  } finally {
    predictButton.disabled = !imageFile || boxes.length === 0;
  }
});

window.addEventListener("resize", () => { fitCanvas(); redraw(); });

// ─── Init ─────────────────────────────────────────────────────────────────────
fitCanvas();
redraw();
syncBoxList();
loadServiceStatus();
