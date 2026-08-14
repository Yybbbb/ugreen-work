const imageInput = document.getElementById("imageInput");
const imageCanvas = document.getElementById("imageCanvas");
const context = imageCanvas.getContext("2d");
const dropZone = document.getElementById("dropZone");
const emptyHint = document.getElementById("emptyHint");
const imageMeta = document.getElementById("imageMeta");
const boxCount = document.getElementById("boxCount");
const boxList = document.getElementById("boxList");
const undoButton = document.getElementById("undoButton");
const clearButton = document.getElementById("clearButton");
const predictButton = document.getElementById("predictButton");
const results = document.getElementById("results");
const resultMeta = document.getElementById("resultMeta");
const callCount = document.getElementById("callCount");
const progressWrap = document.getElementById("progressWrap");
const progressBar = document.getElementById("progressBar");
const serviceStatus = document.getElementById("serviceStatus");

const colors = ["#e83f5b", "#2764d8", "#0f9d78", "#a64bd4", "#ed8b19", "#00a2b8", "#7555d9", "#bd4f16"];
let imageFile = null;
let imageBitmap = null;
let imageWidth = 0;
let imageHeight = 0;
let boxes = [];
let drawing = null;
let inferenceResults = [];

function colorFor(index) { return colors[index % colors.length]; }
function clamp(value, min, max) { return Math.max(min, Math.min(max, value)); }
function normalizeBox(a, b) {
  return {
    x1: clamp(Math.min(a.x, b.x), 0, imageWidth),
    y1: clamp(Math.min(a.y, b.y), 0, imageHeight),
    x2: clamp(Math.max(a.x, b.x), 0, imageWidth),
    y2: clamp(Math.max(a.y, b.y), 0, imageHeight),
  };
}
function isValidBox(box) { return box.x2 - box.x1 >= 4 && box.y2 - box.y1 >= 4; }

function fitCanvas() {
  const availableWidth = Math.max(320, dropZone.clientWidth - 2);
  const scale = Math.min(1, availableWidth / imageWidth);
  imageCanvas.width = Math.round(imageWidth * scale);
  imageCanvas.height = Math.round(imageHeight * scale);
  imageCanvas.style.display = "block";
  emptyHint.style.display = "none";
}

function canvasToImagePoint(event) {
  const rect = imageCanvas.getBoundingClientRect();
  return {
    x: (event.clientX - rect.left) * imageWidth / rect.width,
    y: (event.clientY - rect.top) * imageHeight / rect.height,
  };
}

function drawBox(box, index, temporary = false) {
  const scaleX = imageCanvas.width / imageWidth;
  const scaleY = imageCanvas.height / imageHeight;
  const x = box.x1 * scaleX;
  const y = box.y1 * scaleY;
  const width = (box.x2 - box.x1) * scaleX;
  const height = (box.y2 - box.y1) * scaleY;
  const color = temporary ? "#ffffff" : colorFor(index);
  context.save();
  context.strokeStyle = color;
  context.lineWidth = temporary ? 2 : 3;
  context.setLineDash(temporary ? [7, 5] : []);
  context.strokeRect(x, y, width, height);
  if (!temporary) {
    context.fillStyle = color;
    context.fillRect(x, Math.max(0, y - 24), 34, 24);
    context.fillStyle = "#fff";
    context.font = "700 13px sans-serif";
    context.fillText(`#${index + 1}`, x + 6, Math.max(16, y - 7));
    const description = inferenceResults[index]?.description;
    if (description) {
      const text = description.length > 54 ? `${description.slice(0, 54)}…` : description;
      context.font = "12px sans-serif";
      const textWidth = Math.min(imageCanvas.width - x, context.measureText(text).width + 14);
      context.fillStyle = "rgba(20, 28, 43, .82)";
      context.fillRect(x, Math.min(imageCanvas.height - 24, y + height), textWidth, 24);
      context.fillStyle = "#fff";
      context.fillText(text, x + 7, Math.min(imageCanvas.height - 7, y + height + 17));
    }
  }
  context.restore();
}

function redraw() {
  if (!imageBitmap) return;
  context.clearRect(0, 0, imageCanvas.width, imageCanvas.height);
  context.drawImage(imageBitmap, 0, 0, imageCanvas.width, imageCanvas.height);
  boxes.forEach((box, index) => drawBox(box, index));
  if (drawing?.currentBox) drawBox(drawing.currentBox, boxes.length, true);
}

function updateButtons() {
  const hasImage = Boolean(imageBitmap);
  const hasBoxes = boxes.length > 0;
  undoButton.disabled = !hasBoxes;
  clearButton.disabled = !hasBoxes;
  predictButton.disabled = !(hasImage && hasBoxes);
  boxCount.textContent = `${boxes.length} box${boxes.length === 1 ? "" : "es"}`;
}

function renderBoxList() {
  updateButtons();
  if (!boxes.length) {
    boxList.className = "boxList emptyList";
    boxList.textContent = "暂无区域";
    return;
  }
  boxList.className = "boxList";
  boxList.innerHTML = "";
  boxes.forEach((box, index) => {
    const item = document.createElement("div");
    item.className = "boxItem";
    item.innerHTML = `
      <div class="boxIndex" style="background:${colorFor(index)}">${index + 1}</div>
      <div class="coords">[${Math.round(box.x1)}, ${Math.round(box.y1)}, ${Math.round(box.x2)}, ${Math.round(box.y2)}]</div>
      <button class="deleteButton" type="button">删除</button>`;
    item.querySelector("button").addEventListener("click", () => {
      boxes.splice(index, 1);
      inferenceResults = [];
      renderBoxList();
      clearResults();
      redraw();
    });
    boxList.appendChild(item);
  });
}

function clearResults() {
  inferenceResults = [];
  results.className = "results emptyList";
  results.textContent = "完成推理后，每个框的独立描述会显示在这里。";
  resultMeta.textContent = "等待推理";
  callCount.textContent = "0 calls";
  progressWrap.classList.add("hidden");
  progressBar.style.width = "0";
}

function renderResults() {
  results.className = "results";
  results.innerHTML = "";
  boxes.forEach((box, index) => {
    const result = inferenceResults[index];
    const card = document.createElement("article");
    card.className = `resultCard ${result ? "" : "pending"}`;
    card.innerHTML = `
      <div class="resultTop">
        <span class="boxIndex" style="background:${colorFor(index)}">${index + 1}</span>
        <strong>Region ${index + 1}</strong>
        <span>${result ? `${result.elapsed_ms.toFixed(1)} ms` : "等待调用"}</span>
      </div>
      <p class="description">${result ? escapeHtml(result.description) : "—"}</p>
      <div class="technical">bbox=[${[box.x1, box.y1, box.x2, box.y2].map(Math.round).join(", ")}]</div>
      <div class="technical">${result ? escapeHtml(result.prompt) : ""}</div>`;
    results.appendChild(card);
  });
}

function escapeHtml(text) {
  return String(text).replace(/[&<>'"]/g, char => ({"&":"&amp;","<":"&lt;",">":"&gt;","'":"&#39;",'"':"&quot;"}[char]));
}

async function loadImageFile(file) {
  if (!file || !file.type.startsWith("image/")) throw new Error("请选择有效图片文件");
  imageFile = file;
  imageBitmap = await createImageBitmap(file);
  imageWidth = imageBitmap.width;
  imageHeight = imageBitmap.height;
  boxes = [];
  inferenceResults = [];
  fitCanvas();
  redraw();
  renderBoxList();
  clearResults();
  imageMeta.textContent = `${file.name} · ${imageWidth}×${imageHeight}`;
}

async function predictAll() {
  if (!imageFile || !boxes.length) return;
  predictButton.disabled = true;
  undoButton.disabled = true;
  clearButton.disabled = true;
  inferenceResults = new Array(boxes.length).fill(null);
  renderResults();
  progressWrap.classList.remove("hidden");
  callCount.textContent = `0/${boxes.length} calls`;
  try {
    for (let index = 0; index < boxes.length; index += 1) {
      resultMeta.textContent = `正在调用第 ${index + 1}/${boxes.length} 个单区域请求…`;
      const box = boxes[index];
      const form = new FormData();
      form.append("image", imageFile);
      form.append("bbox", JSON.stringify([box.x1, box.y1, box.x2, box.y2]));
      const response = await fetch("/api/predict-one", { method: "POST", body: form });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || `请求失败：${response.status}`);
      inferenceResults[index] = data;
      progressBar.style.width = `${(index + 1) / boxes.length * 100}%`;
      callCount.textContent = `${index + 1}/${boxes.length} calls`;
      renderResults();
      redraw();
    }
    const totalMs = inferenceResults.reduce((sum, result) => sum + result.elapsed_ms, 0);
    resultMeta.textContent = `完成：${boxes.length} 个框，${boxes.length} 次独立单区域调用，总模型耗时 ${totalMs.toFixed(1)} ms`;
  } catch (error) {
    resultMeta.textContent = error.message;
  } finally {
    updateButtons();
  }
}

async function loadStatus() {
  try {
    const response = await fetch("/api/status");
    const data = await response.json();
    if (!response.ok || !data.ready) throw new Error("模型未就绪");
    serviceStatus.className = "status ready";
    serviceStatus.textContent = `模型已就绪 · ${data.gpu_name} · ${data.precision}`;
  } catch (error) {
    serviceStatus.className = "status error";
    serviceStatus.textContent = `服务异常：${error.message}`;
  }
}

imageInput.addEventListener("change", async () => {
  try { await loadImageFile(imageInput.files?.[0]); }
  catch (error) { resultMeta.textContent = error.message; }
});
dropZone.addEventListener("dragover", event => { event.preventDefault(); dropZone.classList.add("dragActive"); });
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragActive"));
dropZone.addEventListener("drop", async event => {
  event.preventDefault();
  dropZone.classList.remove("dragActive");
  try { await loadImageFile(event.dataTransfer.files?.[0]); }
  catch (error) { resultMeta.textContent = error.message; }
});
imageCanvas.addEventListener("mousedown", event => {
  if (!imageBitmap) return;
  drawing = { start: canvasToImagePoint(event), currentBox: null };
});
imageCanvas.addEventListener("mousemove", event => {
  if (!drawing) return;
  drawing.currentBox = normalizeBox(drawing.start, canvasToImagePoint(event));
  redraw();
});
window.addEventListener("mouseup", () => {
  if (!drawing) return;
  if (drawing.currentBox && isValidBox(drawing.currentBox)) {
    boxes.push(drawing.currentBox);
    inferenceResults = [];
    renderBoxList();
    clearResults();
  }
  drawing = null;
  redraw();
});
undoButton.addEventListener("click", () => {
  boxes.pop(); inferenceResults = []; renderBoxList(); clearResults(); redraw();
});
clearButton.addEventListener("click", () => {
  boxes = []; inferenceResults = []; renderBoxList(); clearResults(); redraw();
});
predictButton.addEventListener("click", predictAll);
window.addEventListener("resize", () => { if (imageBitmap) { fitCanvas(); redraw(); } });

loadStatus();
