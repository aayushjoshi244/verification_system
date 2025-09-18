// index.js
import { GoogleGenerativeAI, SchemaType } from "@google/generative-ai";
import { config } from "dotenv";
import { exec } from "node:child_process";
import { promises as fs } from "node:fs";
import path from "node:path";

// ---------- ADDED: interactive typed input ----------
import readline from "node:readline/promises";
import { stdin as input, stdout as output } from "node:process";
// ----------------------------------------------------

// ---------- ADDED: offline voice support (Vosk + mic) ----------
import vosk from "vosk";                 // npm i vosk
import record from "node-record-lpcm16"; // npm i node-record-lpcm16
// ---------------------------------------------------------------

const USE_SYNTH_FALLBACK = process.env.USE_SYNTH_FALLBACK === "1";

// ---------- ADDED: voice config (env) ----------
const VOICE_MODE = process.env.VOICE_MODE === "1";              // set to "1" to enable voice capture
const VOSK_MODEL_PATH = process.env.VOSK_MODEL_PATH || "./stt-model"; // folder containing Vosk model (e.g., vosk-model-small-en-us-0.15)
const VOICE_SAMPLE_RATE = Number(process.env.VOICE_SAMPLE_RATE || 16000);
const VOICE_TIMEOUT_MS = Number(process.env.VOICE_TIMEOUT_MS || 12000);
// ------------------------------------------------

config();
console.log("📌 CWD at start:", process.cwd());

const GEMINI_API_KEY = process.env.GEMINI_API_KEY;
if (!GEMINI_API_KEY) {
  console.error("❌ GEMINI_API_KEY missing in .env");
  process.exit(1);
}

const client = new GoogleGenerativeAI(GEMINI_API_KEY);

/* --------------------------- Agent Tools ---------------------------------- */
// Prefer safe FS tools over shell:
async function readFileTool(relPath) {
  if (!relPath || !relPath.trim())
    throw new Error("readFile: missing path (e.g., './package.json').");
  const abs = path.resolve(process.cwd(), relPath);
  const data = await fs.readFile(abs, "utf8");
  return data;
}

// batchWriteFiles: input JSON -> { "files": [ { "path": "./index.html", "content": "<!doctype html>..." }, ... ] }
async function batchWriteFilesTool(inputJson) {
  if (!inputJson || !String(inputJson).trim()) {
    throw new Error(
      `batchWriteFiles: missing input. Example: {"files":[{"path":"./index.html","content":"..."}]}`
    );
  }
  let payload;
  try {
    payload = JSON.parse(inputJson);
  } catch {
    throw new Error(
      `batchWriteFiles: input must be JSON. Got: ${String(inputJson).slice(
        0,
        200
      )}`
    );
  }
  const files = payload?.files;
  if (!Array.isArray(files) || files.length === 0) {
    throw new Error(`batchWriteFiles: provide a non-empty "files" array`);
  }
  for (const f of files) {
    if (!f?.path || typeof f.content !== "string") {
      throw new Error(`batchWriteFiles: each file needs {path, content}`);
    }
    const abs = path.resolve(process.cwd(), f.path);
    await fs.mkdir(path.dirname(abs), { recursive: true });
    await fs.writeFile(abs, f.content, "utf8");
  }
  return `✅ wrote ${files.length} file(s): ${files
    .map((f) => f.path)
    .join(", ")}`;
}

async function listDirTool(relPath = ".") {
  const abs = path.resolve(process.cwd(), relPath || ".");
  const entries = await fs.readdir(abs, { withFileTypes: true });
  return JSON.stringify(
    entries.map((e) => ({
      name: e.name,
      type: e.isDirectory() ? "dir" : "file",
    }))
  );
}

async function pwdTool() {
  return process.cwd();
}

// Keep executeCommand if you really need it (guarded)
function executeCommand(command) {
  return new Promise((resolve, reject) => {
    if (!command || !command.trim()) {
      return reject(new Error("executeCommand: missing command string."));
    }
    exec(
      command,
      { timeout: 15_000, maxBuffer: 10 * 1024 * 1024 },
      (error, stdout, stderr) => {
        if (error) {
          return reject(
            new Error(
              `Command failed: ${error.message}\nstdout: ${stdout}\nstderr: ${stderr}`
            )
          );
        }
        resolve(`stdout: ${stdout}\nstderr: ${stderr}`);
      }
    );
  });
}

async function writeFileTool(inputJson) {
  // inputJson is a JSON string: {"path":"./index.html","content":"<html>..."}
  if (!inputJson || !String(inputJson).trim()) {
    throw new Error(
      `writeFile: missing input. Expected JSON string like {"path":"./file","content":"..."} `
    );
  }
  let payload;
  try {
    payload = JSON.parse(inputJson);
  } catch {
    throw new Error(
      `writeFile: input must be a JSON string. Got: ${inputJson.slice(0, 200)}`
    );
  }
  const { path: relPath, content } = payload || {};
  if (!relPath || typeof content !== "string") {
    throw new Error(`writeFile: both "path" and "content" are required.`);
  }
  const abs = path.resolve(process.cwd(), relPath);
  await fs.mkdir(path.dirname(abs), { recursive: true });
  await fs.writeFile(abs, content, "utf8");
  return `✅ wrote ${relPath} (${content.length} chars)`;
}

const TOOLS = {
  readFile: readFileTool,
  listDir: listDirTool,
  pwd: pwdTool,
  executeCommand,
  writeFile: writeFileTool,
  batchWriteFiles: batchWriteFilesTool,
};

/* --------------------------- System Prompt -------------------------------- */
const SYSTEM_PROMPT = `
You are Anantrit, an agentic assistant that solves tasks using a loop of START → THINK → ACTION → OBSERVE → OUTPUT.

Behavior & defaults:
- All file paths are relative to the current working directory (CWD) unless a full path is provided.
- Prefer filesystem tools over shell:
  - Use "pwd" to get the CWD.
  - Use "listDir" to discover files/folders (default path: ".").
  - Use "readFile" to read file contents (e.g., "./package.json").
- Only call "executeCommand" if the user explicitly asks to run a shell command; otherwise avoid it.
- If the user asks about "package.json" and no path is given, assume "./package.json".
- To CREATE multiple files, prefer "batchWriteFiles" (one action for all files).
- For a small website, use a single "batchWriteFiles" with "files": [{path,content}, ...].
- For creation tasks, infer a project folder name from the request (e.g., “weather-dashboard”) and write all files *inside that folder*.
- Never copy example content verbatim. Always generate fresh, request-specific content and structure.

Output contract (STRICT):
- Emit exactly one JSON object per step—no extra prose, no code fences, no surrounding text.
- Valid steps: "think", "action", "observe", "output".
- When "step" is "action":
  - "tool" must be one of ["readFile","listDir","pwd","executeCommand","writeFile","batchWriteFiles"].
  - "input" must be a non-empty string appropriate for that tool (pwd may omit input).
  - DO NOT invent or rename tools.
- If the last step was "observe" with an error message, adjust your plan before the next "action".
- Stop once you can produce a complete answer and emit a single "output" step.

Error-resilient planning:
- If a required argument is missing (e.g., path for readFile), first THINK to pick a sensible default from the context (e.g., "./package.json"), then ACTION with that input.
- Avoid repeating the same ACTION with the same missing/invalid input.
- If a file operation fails (not found/permissions), consider listing the directory, verifying CWD (pwd), or asking for a specific path in your OUTPUT.

Formatting rules:
- JSON only. No backticks, no markdown.
- Keep "content" concise. If including file text, truncate long content (e.g., 10,000 chars) and make it clear that it’s truncated.
- To CREATE or MODIFY files, always use "writeFile" or "batchWriteFiles". Do NOT use shell redirection.
- For "writeFile", put a JSON string in "input", e.g.: {"path":"./index.html","content":"<html>...</html"}.
- For "writeFile", you may provide either:
  (a) "input" as a JSON string: {"path":"./index.html","content":"..."} OR
  (b) "args" as an object: {"path":"./index.html","content":"..."}.
- Do not emit "output" for creation tasks until you have emitted a valid write action and received an OBSERVE confirming success (e.g., message beginning with "✅").
`;

/* --------------------------- JSON Schema ---------------------------------- */
const generationConfig = {
  responseMimeType: "application/json",
  responseSchema: {
    type: SchemaType.OBJECT,
    properties: {
      step: {
        type: SchemaType.STRING,
        enum: ["think", "action", "observe", "output"],
      },
      tool: {
        type: SchemaType.STRING,
        enum: [
          "readFile",
          "listDir",
          "pwd",
          "executeCommand",
          "writeFile",
          "batchWriteFiles",
        ],
        nullable: true,
      },
      input: { type: SchemaType.STRING, nullable: true },
      args: {
        type: SchemaType.OBJECT,
        nullable: true,
        properties: {
          path: { type: SchemaType.STRING, nullable: true },
          content: { type: SchemaType.STRING, nullable: true },
          files: {
            type: SchemaType.ARRAY,
            nullable: true,
            items: {
              type: SchemaType.OBJECT,
              properties: {
                path: { type: SchemaType.STRING },
                content: { type: SchemaType.STRING },
              },
            },
          },
        },
      },
      content: { type: SchemaType.STRING },
    },
    required: ["step", "content"],
  },
};

/* --------------------------- Helpers -------------------------------------- */
function parseToolandInput(toolField, inputField) {
  const toolStr = String(toolField ?? "");
  if (toolStr && inputField != null)
    return { tool: toolStr, input: String(inputField) };
  const m = toolStr.match(/^(\w+)\s*\(\s*([^)]*)\s*\)\s*$/);
  if (m) return { tool: m[1], input: m[2] };
  return { tool: toolStr || "", input: inputField ?? "" };
}

function safeParseJSON(text) {
  if (!text) return null;
  let cleaned = text
    .trim()
    .replace(/^```(?:json)?\s*/i, "")
    .replace(/```$/i, "");
  try {
    return JSON.parse(cleaned);
  } catch {}
  const m = cleaned.match(/\{[\s\S]*\}/);
  if (m) {
    try {
      return JSON.parse(m[0]);
    } catch {}
  }
  return null;
}

function slugify(name) {
  return (
    String(name || "")
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "-")
      .replace(/^-+|-+$/g, "")
      .slice(0, 60) || "project"
  );
}

function inferProjectDir(userQuestion) {
  const q = String(userQuestion || "");

  // explicit patterns
  const m1 = q.match(
    /\b(?:named|called|name it|folder(?: name)?)\s+["'“”]?([A-Za-z0-9 _-]{2,})["'“”]?/i
  );
  if (m1?.[1]) return slugify(m1[1]);

  const m2 = q.match(
    /\b(?:create|build|make|generate)\b.*?\b([A-Za-z0-9_-]{3,})\s*(?:app|project|site|website)\b/i
  );
  if (m2?.[1]) return slugify(m2[1]);

  const m3 = q.match(/\b([A-Za-z0-9_-]{3,})\s*(?:app|project|site|website)\b/i);
  if (m3?.[1]) return slugify(m3[1]);

  // fallback
  return "project";
}

function ensureUnderBaseDir(relPath, baseDir) {
  const p = String(relPath || "");
  // If there is no directory component, or it starts with "./" only, attach baseDir
  const hasSlashBeyondDot = /\/.+/.test(p.replace(/^\.\//, ""));
  if (!hasSlashBeyondDot) {
    return `./${baseDir}/${p.replace(/^\.\//, "")}`;
  }
  return p;
}

function rewriteBatchInputToBaseDir(inputJson, baseDir) {
  let payload = JSON.parse(inputJson);
  if (!payload || !Array.isArray(payload.files)) return inputJson;
  payload = {
    ...payload,
    files: payload.files.map((f) => ({
      ...f,
      path: ensureUnderBaseDir(f.path, baseDir),
    })),
  };
  return JSON.stringify(payload);
}

function rewriteSingleWriteToBaseDir(inputJson, baseDir) {
  let payload = JSON.parse(inputJson);
  if (!payload || !payload.path) return inputJson;
  payload = { ...payload, path: ensureUnderBaseDir(payload.path, baseDir) };
  return JSON.stringify(payload);
}

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}
function isRetryableError(err) {
  const code = err?.status;
  const msg = String(err?.message || "");
  return (
    code === 429 ||
    code === 500 ||
    code === 502 ||
    code === 503 ||
    code === 504 ||
    /ECONNRESET|ETIMEDOUT|ENOTFOUND|EAI_AGAIN|socket hang up/i.test(msg)
  );
}
async function generateWithRetry(
  model,
  args,
  { maxRetries = 5, initialDelayMs = 400, maxDelayMs = 6000 } = {}
) {
  let delay = initialDelayMs;
  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      return await model.generateContent(args);
    } catch (err) {
      if (attempt === maxRetries || !isRetryableError(err)) throw err;
      const jitter = Math.min(
        maxDelayMs,
        Math.round(delay * (1 + Math.random()))
      );
      console.warn(
        `↻ retry ${attempt + 1}: ${err?.status || ""} waiting ${jitter}ms`
      );
      await sleep(jitter);
      delay = Math.min(maxDelayMs, delay * 2);
    }
  }
}

/* --------------------- Local Fast-Paths: Apps ----------------------------- */
/** Detects requests that ask for a todo app to be created. */
function isTodoRequest(text) {
  if (!text) return false;
  const q = text.toLowerCase();
  return (
    /\b(todo|to-do)\b/.test(q) &&
    /\b(app|application|site|website|project)\b/.test(q) &&
    /\b(create|build|make|generate)\b/.test(q)
  );
}

/** Detects requests that ask for a weather app to be created. */
function isWeatherRequest(text) {
  if (!text) return false;
  const q = text.toLowerCase();
  return (
    /\b(weather|wheather|meteo|forecast)\b/.test(q) &&
    /\b(app|application|site|website|project|dashboard)\b/.test(q) &&
    /\b(create|build|make|generate)\b/.test(q)
  );
}

/** Returns high-quality files for a complete Todo app (no external libs). */
function buildTodoFiles(baseDir) {
  const indexHtml = `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Todo — Minimal & Fast</title>
  <link rel="stylesheet" href="./styles.css">
  <link rel="icon" href="data:,">
</head>
<body>
  <main class="wrap" aria-labelledby="app-title">
    <header class="header">
      <h1 id="app-title">Tasks</h1>
      <form id="new-form" aria-label="Add new task">
        <input id="new-input" type="text" placeholder="What needs to be done?"
               autocomplete="off" aria-label="Task title" />
        <button id="add-btn" type="submit" aria-label="Add task">Add</button>
      </form>
    </header>

    <section class="controls">
      <div class="filters" role="tablist" aria-label="Filters">
        <button class="filter is-active" data-filter="all" role="tab" aria-selected="true">All</button>
        <button class="filter" data-filter="active" role="tab" aria-selected="false">Active</button>
        <button class="filter" data-filter="completed" role="tab" aria-selected="false">Completed</button>
      </div>
      <div class="meta">
        <span id="left-count" aria-live="polite">0 items left</span>
        <button id="clear-completed" class="link">Clear completed</button>
      </div>
    </section>

    <ul id="list" class="list" aria-label="Todo list"></ul>

    <footer class="footer">
      <p>Drag to reorder • Double-click a task to edit • Data persists locally</p>
    </footer>
  </main>
  <script src="./script.js"></script>
</body>
</html>`;

  const stylesCss = `:root{
  --bg:#0f1115;--elev:#151823;--text:#e8ebf0;--muted:#a9b3c7;--brand:#6cf;--ok:#8de26a;--warn:#ffcf5a;--danger:#ff6b6b;--ring:#6cf7;
  --radius:16px;--pad:18px;--gap:14px;--shadow:0 10px 30px rgba(0,0,0,.35),0 1px 0 rgba(255,255,255,.03) inset;
}
*{box-sizing:border-box}
html,body{margin:0;height:100%}
body{font:16px/1.5 system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,Cantarell,Noto Sans,sans-serif;background:radial-gradient(1200px 600px at 10% -10%,#1a2040 0,#0f1115 60%),var(--bg);color:var(--text)}
.wrap{max-width:860px;margin:40px auto;padding:0 20px}
.header{background:linear-gradient(180deg,#1b2140 0,#141827 100%);border:1px solid #1f2540;border-radius:var(--radius);padding:24px;box-shadow:var(--shadow)}
.header h1{margin:0 0 14px;font-weight:800;letter-spacing:.5px;font-size:28px;background:linear-gradient(90deg,#b2c7ff,#6cf);-webkit-background-clip:text;background-clip:text;color:transparent}
#new-form{display:flex;gap:10px}
#new-input{flex:1;padding:12px 14px;border-radius:12px;border:1px solid #28304d;background:#0e1220;color:var(--text);outline:none}
#new-input:focus{box-shadow:0 0 0 3px rgba(102,204,255,.25)}
#add-btn{padding:12px 16px;border-radius:12px;border:0;background:linear-gradient(90deg,#37a6ff,#38e8ff);color:#062134;font-weight:700;cursor:pointer}
#add-btn:active{transform:translateY(1px)}

.controls{display:flex;align-items:center;justify-content:space-between;margin:18px 4px}
.filters{display:flex;gap:8px;padding:6px;background:#0e1220;border:1px solid #28304d;border-radius:999px}
.filter{padding:8px 12px;border-radius:999px;border:0;background:transparent;color:var(--muted);cursor:pointer}
.filter.is-active{background:#1a2036;color:var(--text);box-shadow:inset 0 0 0 1px #2a345a}

.meta{display:flex;gap:14px;align-items:center;color:var(--muted);font-size:14px}
.link{padding:6px 10px;border-radius:10px;border:1px solid #2a345a;background:#101628;color:#b9c3da;cursor:pointer}
.link:hover{border-color:#3a4870}

.list{list-style:none;margin:0;padding:0;display:flex;flex-direction:column;gap:8px}
.item{display:grid;grid-template-columns:auto 1fr auto;align-items:center;gap:12px;padding:12px 14px;background:linear-gradient(180deg,#121728,#0d1120);border:1px solid #242d4a;border-radius:14px;box-shadow:var(--shadow)}
.item.dragging{opacity:.6}
.chk{appearance:none;width:20px;height:20px;border:2px solid #38507a;border-radius:6px;display:inline-grid;place-items:center;cursor:pointer;background:#0c1222}
.chk:checked{background:linear-gradient(90deg,#24d3ff,#64f1a9);border-color:transparent}
.chk:checked::after{content:"";width:10px;height:10px;background:#07121a;border-radius:3px}
.text{color:var(--text)}
.text.completed{color:#8aa0be;text-decoration:line-through}
.actions{display:flex;gap:6px}
.btn{padding:8px 10px;border-radius:10px;border:1px solid #2a345a;background:#0f1426;color:#cbd6f0;cursor:pointer}
.btn:hover{border-color:#3a4b78}
.btn.danger{border-color:#4b2a2a;background:#1a0f0f;color:#ffc0c0}
.edit{width:100%;padding:10px 12px;border-radius:10px;border:1px solid #3a4b78;background:#0e1220;color:var(--text);outline:none}

.footer{margin:22px 4px;color:#94a3c7;font-size:13px;text-align:center}
@media (max-width:640px){
  .header{padding:18px}
  #add-btn{padding:12px}
}
`;

  const scriptJs = `(() => {
  const KEY = "todo-items-v1";
  let items = load();
  let filter = "all";
  const listEl = document.getElementById("list");
  const newForm = document.getElementById("new-form");
  const newInput = document.getElementById("new-input");
  const leftCount = document.getElementById("left-count");
  const clearBtn = document.getElementById("clear-completed");
  const filterButtons = Array.from(document.querySelectorAll(".filter"));
  let draggingId = null;

  function save() { localStorage.setItem(KEY, JSON.stringify(items)); }
  function load() {
    try { return JSON.parse(localStorage.getItem(KEY) || "[]"); } catch { return []; }
  }
  function uid() { return Math.random().toString(36).slice(2, 10); }
  function activeCount() { return items.filter(i => !i.completed).length; }
  function nextOrder() { return items.length ? Math.max(...items.map(i => i.order)) + 1 : 1; }

  function render() {
    listEl.innerHTML = "";
    const filtered = items.slice().sort((a,b) => a.order - b.order)
      .filter(i => filter === "all" ? true : filter === "active" ? !i.completed : i.completed);
    for (const it of filtered) listEl.appendChild(renderItem(it));
    const n = activeCount();
    leftCount.textContent = n === 1 ? "1 item left" : n + " items left";
    filterButtons.forEach(b => {
      const sel = b.dataset.filter === filter;
      b.classList.toggle("is-active", sel);
      b.setAttribute("aria-selected", String(sel));
    });
  }

  function renderItem(it) {
    const li = document.createElement("li"); li.className="item"; li.draggable=true; li.dataset.id=it.id;
    const chk = Object.assign(document.createElement("input"), {type:"checkbox", className:"chk", checked:it.completed});
    chk.addEventListener("change", ()=>{ it.completed = chk.checked; save(); render(); });

    const text = document.createElement("div"); text.className = "text" + (it.completed ? " completed": ""); text.textContent = it.text;
    text.addEventListener("dblclick", ()=> startEdit(li, it));

    const actions = document.createElement("div"); actions.className="actions";
    const editBtn = document.createElement("button"); editBtn.className="btn"; editBtn.textContent="Edit";
    editBtn.addEventListener("click", ()=> startEdit(li, it));
    const delBtn = document.createElement("button"); delBtn.className="btn danger"; delBtn.textContent="Delete";
    delBtn.addEventListener("click", ()=>{ items = items.filter(x=>x.id!==it.id); save(); render(); });
    actions.append(editBtn, delBtn);

    li.append(chk, text, actions);

    li.addEventListener("dragstart", e => {
      draggingId = it.id; li.classList.add("dragging");
      e.dataTransfer?.setData("text/plain", it.id);
      e.dataTransfer?.setDragImage(new Image(), 0, 0);
    });
    li.addEventListener("dragend", ()=>{ draggingId=null; li.classList.remove("dragging"); });
    li.addEventListener("dragover", e => {
      e.preventDefault();
      const after = getDragAfterElement(document.getElementById("list"), e.clientY);
      const draggingEl = document.querySelector(".item.dragging");
      if (!draggingEl) return;
      if (!after) listEl.appendChild(draggingEl); else listEl.insertBefore(draggingEl, after);
    });
    li.addEventListener("drop", ()=>{ applyVisualOrderToItems(); save(); render(); });
    return li;
  }

  function getDragAfterElement(container, y) {
    const els = [...container.querySelectorAll(".item:not(.dragging)")];
    return els.reduce((closest, child) => {
      const box = child.getBoundingClientRect();
      const offset = y - box.top - box.height/2;
      if (offset < 0 && offset > closest.offset) return {offset, element:child};
      else return closest;
    }, {offset: Number.NEGATIVE_INFINITY, element:null}).element;
  }
  function applyVisualOrderToItems() {
    const ids = [...document.querySelectorAll(".item")].map(li => li.dataset.id);
    ids.forEach((id, idx)=>{ const it = items.find(x=>x.id===id); if (it) it.order = (idx+1)*10; });
  }

  function startEdit(li, it) {
    const textEl = li.querySelector(".text");
    const input = document.createElement("input"); input.className="edit"; input.type="text"; input.value = it.text;
    textEl.replaceWith(input); input.focus(); input.setSelectionRange(input.value.length, input.value.length);
    const commit = ()=>{ const v = input.value.trim(); it.text = v || it.text; save(); render(); };
    const cancel = ()=> render();
    input.addEventListener("keydown", e=>{ if (e.key==="Enter") commit(); else if (e.key==="Escape") cancel(); });
    input.addEventListener("blur", commit, {once:true});
  }

  document.getElementById("new-form").addEventListener("submit", e=>{
    e.preventDefault();
    const v = newInput.value.trim(); if (!v) return;
    items.push({ id: Math.random().toString(36).slice(2,10), text: v, completed:false, order: nextOrder() });
    newInput.value=""; save(); render();
  });
  document.getElementById("clear-completed").addEventListener("click", ()=>{
    items = items.filter(i=>!i.completed); save(); render();
  });
  document.querySelectorAll(".filter").forEach(btn=>{
    btn.addEventListener("click", ()=>{ filter = btn.dataset.filter || "all"; render(); });
  });

  render();
})();`;

  return [
    { path: `./${baseDir}/index.html`, content: indexHtml },
    { path: `./${baseDir}/styles.css`, content: stylesCss },
    { path: `./${baseDir}/script.js`, content: scriptJs },
  ];
}

/** Returns files for a Weather app using Open-Meteo APIs (no key required). */
function buildWeatherFiles(baseDir) {
  const indexHtml = `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Weather — 7-Day Forecast</title>
  <link rel="stylesheet" href="./styles.css" />
  <link rel="icon" href="data:,">
</head>
<body>
  <main class="wrap" aria-labelledby="title">
    <header class="header">
      <h1 id="title">Weather</h1>
      <form id="search-form" aria-label="Search city">
        <input id="city-input" type="text" placeholder="Enter city (e.g., Seoul)" autocomplete="off" aria-label="City name" />
        <button id="go" type="submit">Search</button>
      </form>
      <button id="use-geo" class="link" aria-label="Use my location">Use my location</button>
    </header>

    <section class="status" id="status" aria-live="polite"></section>

    <section id="current" class="current" hidden>
      <div class="now">
        <div class="now-temp" id="now-temp">--°</div>
        <div class="now-city" id="now-city">—</div>
        <div class="now-desc" id="now-desc">—</div>
      </div>
      <div class="now-meta">
        <div><span>Wind</span><strong id="now-wind">—</strong></div>
        <div><span>Humidity</span><strong id="now-hum">—</strong></div>
        <div><span>Feels</span><strong id="now-feels">—</strong></div>
      </div>
    </section>

    <section id="daily" class="daily" hidden aria-label="7-day forecast"></section>

    <footer class="footer"><p>Data: Open-Meteo • No key required • Caching last city locally</p></footer>
  </main>
  <script src="./script.js"></script>
</body>
</html>`;

  const stylesCss = `:root{
  --bg:#0f1115;--panel:#141a27;--text:#e8eef9;--muted:#9fb0cf;--brand:#6cf;--ring:#66ccff55;
  --hot:#ff8a66;--mild:#ffd166;--cool:#66d1ff;--cold:#9ad0ff;
  --radius:16px;--shadow:0 10px 30px rgba(0,0,0,.35),0 1px 0 rgba(255,255,255,.03) inset;
}
*{box-sizing:border-box}
html,body{margin:0;height:100%}
body{font:16px/1.5 system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,Cantarell,Noto Sans,sans-serif;background:radial-gradient(1200px 600px at 10% -10%,#1a2040 0,#0f1115 60%),var(--bg);color:var(--text)}
.wrap{max-width:980px;margin:40px auto;padding:0 20px}
.header{background:linear-gradient(180deg,#1b2140 0,#141827 100%);border:1px solid #1f2540;border-radius:var(--radius);padding:24px;box-shadow:var(--shadow);display:grid;gap:12px}
.header h1{margin:0;font-weight:800;letter-spacing:.5px;font-size:28px;background:linear-gradient(90deg,#b2c7ff,#6cf);-webkit-background-clip:text;background-clip:text;color:transparent}
#search-form{display:flex;gap:10px}
#city-input{flex:1;padding:12px 14px;border-radius:12px;border:1px solid #28304d;background:#0e1220;color:var(--text);outline:none}
#city-input:focus{box-shadow:0 0 0 3px var(--ring)}
#go{padding:12px 16px;border-radius:12px;border:0;background:linear-gradient(90deg,#37a6ff,#38e8ff);color:#062134;font-weight:700;cursor:pointer}
.link{justify-self:start;padding:8px 10px;border-radius:10px;border:1px solid #2a345a;background:#101628;color:#b9c3da;cursor:pointer}

.status{margin:16px 4px;color:var(--muted);min-height:1.2em}
.current{display:grid;gap:16px;margin:14px 0;padding:18px;border:1px solid #253054;border-radius:16px;background:linear-gradient(180deg,#121728,#0e1323);box-shadow:var(--shadow)}
.now{display:flex;align-items:baseline;gap:14px}
.now-temp{font-size:48px;font-weight:800}
.now-city{font-size:20px;color:#cfe0ff}
.now-desc{color:#a7b9dc}
.now-meta{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:10px}
.now-meta div{padding:10px;border-radius:12px;background:#101629;border:1px solid #223050}
.now-meta span{display:block;font-size:12px;color:#a5b1c9}
.now-meta strong{font-size:16px}

.daily{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px}
.card{padding:14px;border-radius:14px;background:#0f1528;border:1px solid #243357;box-shadow:var(--shadow)}
.card h3{margin:0 0 6px;font-size:14px;color:#cfe0ff}
.card .temp{font-weight:800;font-size:22px;margin:6px 0}
.band{height:6px;border-radius:6px;background:linear-gradient(90deg,var(--cold),var(--cool),var(--mild),var(--hot))}

.footer{margin:22px 4px;color:#94a3c7;font-size:13px;text-align:center}
@media (max-width:640px){
  .header{padding:18px}
  #go{padding:12px}
}
`;

  const scriptJs = `(() => {
  const GEO_URL = "https://geocoding-api.open-meteo.com/v1/search?count=1&language=en&format=json&name=";
  const METEO_URL = "https://api.open-meteo.com/v1/forecast";
  const LAST_KEY = "weather-last-city";

  const statusEl = document.getElementById("status");
  const cityInput = document.getElementById("city-input");
  const form = document.getElementById("search-form");
  const btnGeo = document.getElementById("use-geo");
  const current = {
    box: document.getElementById("current"),
    temp: document.getElementById("now-temp"),
    city: document.getElementById("now-city"),
    desc: document.getElementById("now-desc"),
    wind: document.getElementById("now-wind"),
    hum: document.getElementById("now-hum"),
    feels: document.getElementById("now-feels"),
  };
  const dailyEl = document.getElementById("daily");

  function setStatus(msg) { statusEl.textContent = msg || ""; }
  function setError(msg) { statusEl.textContent = "⚠️ " + msg; }

  form.addEventListener("submit", async (e)=>{
    e.preventDefault();
    const name = (cityInput.value || "").trim();
    if (!name) { setError("Enter a city name."); return; }
    await runByCity(name);
  });

  btnGeo.addEventListener("click", ()=>{
    if (!navigator.geolocation) { setError("Geolocation not supported by your browser."); return; }
    setStatus("Detecting your location…");
    navigator.geolocation.getCurrentPosition(async pos=>{
      try {
        const {latitude:lat, longitude:lon} = pos.coords;
        await runByCoords(lat, lon, "My location");
      } catch (e) { setError("Could not load weather for your location."); }
    }, err => setError("Location error: " + (err?.message || "unknown")));
  });

  async function runByCity(name) {
    try {
      setStatus("Looking up city…");
      const geoRes = await fetch(GEO_URL + encodeURIComponent(name));
      if (!geoRes.ok) throw new Error("Geocoding failed (" + geoRes.status + ")");
      const geo = await geoRes.json();
      const hit = geo?.results?.[0];
      if (!hit) { setError("City not found."); return; }
      const fullName = [hit.name, hit.admin1, hit.country].filter(Boolean).join(", ");
      localStorage.setItem(LAST_KEY, fullName);
      await runByCoords(hit.latitude, hit.longitude, fullName);
    } catch (e) {
      setError(e.message || "Error fetching city.");
    }
  }

  async function runByCoords(lat, lon, label) {
    try {
      setStatus("Fetching weather…");
      const url = METEO_URL + "?latitude=" + lat + "&longitude=" + lon + "&current=temperature_2m,relative_humidity_2m,apparent_temperature,weather_code,wind_speed_10m&daily=weather_code,temperature_2m_max,temperature_2m_min&timezone=auto";
      const res = await fetch(url);
      if (!res.ok) throw new Error("Forecast failed (" + res.status + ")");
      const data = await res.json();
      paintCurrent(data, label);
      paintDaily(data);
      setStatus("");
    } catch (e) {
      setError(e.message || "Error fetching forecast.");
    }
  }

  function paintCurrent(d, label) {
    current.box.hidden = false;
    const c = d.current;
    current.temp.textContent = Math.round(c.temperature_2m) + "°";
    current.city.textContent = label;
    current.desc.textContent = describeCode(c.weather_code);
    current.wind.textContent = Math.round(c.wind_speed_10m) + " km/h";
    current.hum.textContent = Math.round(c.relative_humidity_2m) + "%";
    current.feels.textContent = Math.round(c.apparent_temperature) + "°";
  }

  function paintDaily(d) {
    dailyEl.hidden = false;
    dailyEl.innerHTML = "";
    const days = d.daily.time;
    for (let i=0; i<days.length; i++) {
      const date = new Date(days[i]);
      const max = Math.round(d.daily.temperature_2m_max[i]);
      const min = Math.round(d.daily.temperature_2m_min[i]);
      const code = d.daily.weather_code[i];
      const card = document.createElement("div");
      card.className = "card";
      const title = document.createElement("h3");
      title.textContent = date.toLocaleDateString(undefined, { weekday: "short", month: "short", day: "numeric" });
      const desc = document.createElement("div");
      desc.textContent = describeCode(code);
      const temp = document.createElement("div");
      temp.className = "temp";
      temp.textContent = max + "° / " + min + "°";
      const band = document.createElement("div");
      band.className = "band";
      card.append(title, desc, temp, band);
      dailyEl.appendChild(card);
    }
  }

  // Simple WMO code mapping (subset)
  function describeCode(code) {
    const map = {
      0:"Clear sky", 1:"Mainly clear", 2:"Partly cloudy", 3:"Overcast",
      45:"Fog", 48:"Depositing rime fog",
      51:"Light drizzle", 53:"Moderate drizzle", 55:"Dense drizzle",
      61:"Light rain", 63:"Moderate rain", 65:"Heavy rain",
      71:"Light snow", 73:"Moderate snow", 75:"Heavy snow",
      80:"Rain showers", 81:"Rain showers", 82:"Violent rain showers",
      95:"Thunderstorm", 96:"Thunderstorm w/ hail", 99:"Severe thunderstorm"
    };
    return map[code] || "—";
  }

  // Restore last searched city on load
  const last = localStorage.getItem(LAST_KEY);
  if (last) { cityInput.value = last; runByCity(last); }
})();`;

  return [
    { path: `./${baseDir}/index.html`, content: indexHtml },
    { path: `./${baseDir}/styles.css`, content: stylesCss },
    { path: `./${baseDir}/script.js`, content: scriptJs },
  ];
}

/* --------------------------- Agent Loop ----------------------------------- */
async function runWorkflow(userQuestion, { maxSteps = 200 } = {}) {
  // ✅ Local fast-paths BEFORE any Gemini calls:
  const baseDirEarly = inferProjectDir(userQuestion);

  if (isTodoRequest(userQuestion)) {
    const files = buildTodoFiles(baseDirEarly);
    const inputJson = JSON.stringify({ files });
    const observed = await TOOLS.batchWriteFiles(inputJson);
    const folder = path.resolve(process.cwd(), baseDirEarly);
    const written = await fs.readdir(folder).catch(() => []);
    return `✅ Todo app created in ${folder}:\n- ${written.join("\n- ")}\n\n${observed}`;
  }

  if (isWeatherRequest(userQuestion)) {
    const files = buildWeatherFiles(baseDirEarly);
    const inputJson = JSON.stringify({ files });
    const observed = await TOOLS.batchWriteFiles(inputJson);
    const folder = path.resolve(process.cwd(), baseDirEarly);
    const written = await fs.readdir(folder).catch(() => []);
    return `✅ Weather app created in ${folder}:\n- ${written.join("\n- ")}\n\n${observed}`;
  }

  // Default (agentic) flow using Gemini:
  const model = client.getGenerativeModel({
    model: "gemini-1.5-flash",
    systemInstruction: SYSTEM_PROMPT,
  });

  const contents = [{ role: "user", parts: [{ text: userQuestion }] }];

  const baseDir = baseDirEarly;
  console.log("📂 Base project dir:", baseDir);

  let lastActionSig = "";
  let sameActionCount = 0;
  const MAX_SAME_ACTION = 3;
  let didWriteAnything = false;

  for (let i = 0; i < maxSteps; i++) {
    const res = await generateWithRetry(model, { contents, generationConfig });
    const text = res.response.text();
    const step = safeParseJSON(text);

    if (!step) {
      contents.push({
        role: "user",
        parts: [
          {
            text: `{"step":"continue","content":"Re-emit a single valid JSON step."}`,
          },
        ],
      });
      continue;
    }

    console.log("Anantrit:", JSON.stringify(step));
    contents.push({ role: "model", parts: [{ text: JSON.stringify(step) }] });

    const tag = String(step.step || "").toLowerCase();

    if (tag === "output") {
      const creationIntent = /create|build|generate|make/i.test(userQuestion);
      if (creationIntent && !didWriteAnything) {
        const observeObj = {
          step: "observe",
          tool: "",
          input: "",
          content:
            "You cannot emit an output yet—no files were written. Now produce ONE action with tool=batchWriteFiles and a non-empty 'input' JSON. Your file CONTENT must be tailored to this request and different from any examples.",
        };
        contents.push({ role: "user", parts: [{ text: JSON.stringify(observeObj) }] });
        continue;
      }
      return step.content;
    }

    if (tag === "action") {
      const { tool, input } = parseToolandInput(step.tool, step.input);
      let fixedInput = input;

      if (tool === "writeFile" && fixedInput && String(fixedInput).trim()) {
        try { fixedInput = rewriteSingleWriteToBaseDir(fixedInput, baseDir); } catch {}
      }
      if (tool === "batchWriteFiles" && fixedInput && String(fixedInput).trim()) {
        try { fixedInput = rewriteBatchInputToBaseDir(fixedInput, baseDir); } catch {}
      }

      const sig = JSON.stringify({ tool, hasInput: !!(fixedInput && String(fixedInput).trim()) });
      if (sig === lastActionSig) sameActionCount++; else { sameActionCount = 0; lastActionSig = sig; }

      if (tool === "writeFile" && !fixedInput && step.args?.path && step.args?.content) {
        fixedInput = JSON.stringify({ path: step.args.path, content: step.args.content });
        try { fixedInput = rewriteSingleWriteToBaseDir(fixedInput, baseDir); } catch {}
      }

      if (tool === "batchWriteFiles" && !fixedInput && Array.isArray(step.args?.files)) {
        fixedInput = JSON.stringify({ files: step.args.files });
        try { fixedInput = rewriteBatchInputToBaseDir(fixedInput, baseDir); } catch {}
      }

      if (tool === "batchWriteFiles" && (!fixedInput || !String(fixedInput).trim())) {
        if (sameActionCount >= MAX_SAME_ACTION) {
          if (!USE_SYNTH_FALLBACK) {
            const example = {
              step: "action",
              tool: "batchWriteFiles",
              input: JSON.stringify({
                files: [
                  { path: `./${baseDir}/index.html`, content: "<!doctype html>...</html>" },
                  { path: `./${baseDir}/styles.css`, content: "/* css */" },
                  { path: `./${baseDir}/script.js`, content: "// js" },
                ],
              }),
              content: "Create project files",
            };
            const observeObj = {
              step: "observe",
              tool: "batchWriteFiles",
              input: "",
              content:
                'batchWriteFiles error: missing input. Provide JSON like {"files":[{"path":"./file","content":"..."}]}.\n' +
                "Emit EXACTLY ONE action next with your own generated file contents, and all paths under ./" + baseDir + ".\n" +
                JSON.stringify(example),
            };
            contents.push({ role: "user", parts: [{ text: JSON.stringify(observeObj) }] });
            continue;
          } else {
            const title = baseDir.replace(/-/g, " ");
            const synthesized = {
              files: [
                { path: `./${baseDir}/index.html`, content: '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+title+'</title><link rel="stylesheet" href="./styles.css"></head><body><div id="app"></div><script src="./script.js"></script></body></html>' },
                { path: `./${baseDir}/styles.css`, content: "html,body{margin:0;padding:0;font-family:system-ui}#app{max-width:960px;margin:0 auto;padding:2rem}\n" },
                { path: `./${baseDir}/script.js`, content: 'document.getElementById("app").innerHTML = "<h1>'+title+'</h1>";\n' },
              ],
            };
            const inputJson = JSON.stringify(synthesized);
            try {
              const observed = await TOOLS.batchWriteFiles(inputJson);
              const folder = path.resolve(process.cwd(), baseDir);
              const files = await fs.readdir(folder).catch(() => []);
              console.log(`🔧 batchWriteFiles(synthesized) -> ${observed}`);
              console.log(`📁 WROTE FILES under ./${baseDir}:`, files);
              return `✅ Created scaffold in ${folder}:\n- ${files.join("\n- ")}`;
            } catch (e) {
              contents.push({ role: "user", parts: [{ text: JSON.stringify({ step: "observe", tool: "batchWriteFiles", input: inputJson, content: `Tool batchWriteFiles error: ${e?.message || e}` }) }] });
              continue;
            }
          }
        }
      }

      if (!tool || !(tool in TOOLS)) {
        contents.push({ role: "user", parts: [{ text: JSON.stringify({ step: "observe", tool, input: fixedInput, content: `Unsupported tool: ${tool || "(none)"}` }) }] });
        continue;
      }

      if (tool === "readFile") {
        if (!fixedInput || !String(fixedInput).trim()) {
          const askedForPkg = /package\.json/i.test(userQuestion);
          if (askedForPkg) fixedInput = "./package.json";
        }
        if (!fixedInput || !String(fixedInput).trim()) {
          contents.push({ role: "user", parts: [{ text: JSON.stringify({ step: "observe", tool, input: fixedInput, content: 'readFile error: missing path. Example: "./package.json".' }) }] });
          continue;
        }
      }

      if (tool === "listDir") {
        if (!fixedInput || !String(fixedInput).trim()) fixedInput = ".";
      }

      if (tool === "writeFile") {
        if (!fixedInput || !String(fixedInput).trim()) {
          const template = {
            step: "action",
            tool: "writeFile",
            input: JSON.stringify({ path: `./${baseDir}/index.html`, content: "<!-- content omitted -->" }),
            content: "Create index.html",
          };
          contents.push({ role: "user", parts: [{ text: JSON.stringify({ step: "observe", tool, input: fixedInput, content: 'writeFile error: missing input. Provide JSON string like {"path":"./file","content":"..."}. Next step like:\n' + JSON.stringify(template) }) }] });
          continue;
        }
      }

      if (tool === "executeCommand") {
        if ((!fixedInput || !String(fixedInput).trim()) && typeof step.content === "string") {
          const looksLikeCmd = /^[\s]*(echo|printf|type|cat|powershell|cmd|npm|npx|mkdir|dir|ls|touch)\b/i.test(step.content);
          if (looksLikeCmd) fixedInput = step.content.trim();
        }
        if (!fixedInput || !String(fixedInput).trim()) {
          contents.push({ role: "user", parts: [{ text: JSON.stringify({ step: "observe", tool, input: fixedInput, content: "Missing input for executeCommand. Provide a full command string." }) }] });
          continue;
        }
      }

      let observed;
      try {
        observed = await TOOLS[tool](fixedInput);
        if ((tool === "writeFile" || tool === "batchWriteFiles") && typeof observed === "string" && observed.startsWith("✅")) {
          didWriteAnything = true;
          const files = await fs.readdir(`./${baseDir}`).catch(() => []);
          console.log(`📁 WROTE FILES under ./${baseDir}:`, files);
        }
      } catch (e) {
        observed = `Tool ${tool} error: ${e?.message || e}`;
      }

      console.log(`🔧 ${tool}(${fixedInput ?? ""}) -> ${String(observed).slice(0, 200)}${String(observed).length > 200 ? "…" : ""}`);
      contents.push({ role: "user", parts: [{ text: JSON.stringify({ step: "observe", tool, input: fixedInput, content: String(observed) }) }] });
      continue;
    }

    contents.push({ role: "user", parts: [{ text: `{"step":"continue","content":"Proceed to the next step."}` }] });
  }

  if (didWriteAnything) {
    const folder = path.resolve(process.cwd(), inferProjectDir(userQuestion));
    const files = await fs.readdir(folder).catch(() => []);
    return `✅ Files created in ${folder}:\n- ${files.join("\n- ")}`;
  }
  throw new Error("Max steps reached without OUTPUT");
}

/* --------------------------- Voice capture (unchanged) -------------------- */
async function listenForVoiceCommand({
  modelPath = VOSK_MODEL_PATH,
  sampleRate = VOICE_SAMPLE_RATE,
  timeoutMs = VOICE_TIMEOUT_MS,
} = {}) {
  try {
    const st = await fs.stat(modelPath);
    if (!st.isDirectory()) throw new Error("not a directory");
  } catch {
    console.error(`❌ Vosk model not found at "${modelPath}". Set VOSK_MODEL_PATH or disable VOICE_MODE.`);
    return "";
  }

  vosk.setLogLevel(0);
  const model = new vosk.Model(modelPath);
  const rec = new vosk.Recognizer({ model, sampleRate });

  console.log("🎙️  Speak your command (e.g., “create a todo app” or “create a weather app”)…");

  const mic = record.record({
    sampleRateHertz: sampleRate,
    threshold: 0,
    verbose: false,
    recordProgram: process.platform === "win32" ? "sox" : "rec",
    audioType: "wav",
    endOnSilence: true,
    silence: "1.0",
  });

  const stream = mic.stream();
  let finalText = "";
  let timedOut = false;

  const tm = setTimeout(() => {
    timedOut = true;
    try { mic.stop(); } catch {}
  }, timeoutMs);

  stream.on("data", (data) => {
    rec.acceptWaveform(data);
  });

  const endPromise = new Promise((resolve) => {
    stream.on("end", () => resolve());
    stream.on("close", () => resolve());
    stream.on("error", () => resolve());
  });

  await endPromise;
  clearTimeout(tm);

  try {
    const res = rec.finalResult(); // { text: "..." }
    finalText = (res && res.text) ? String(res.text).trim() : "";
  } catch {}
  rec.free();
  model.free();

  if (!finalText && timedOut) {
    console.log("⌛ Voice capture timeout. No speech recognized.");
  } else {
    console.log("📝 Recognized:", finalText || "(empty)");
  }
  return finalText;
}

/* --------------------------- ADDED: prompt helpers ------------------------ */
async function promptOnce(message) {
  const rl = readline.createInterface({ input, output });
  const ans = await rl.question(message);
  await rl.close();
  return (ans || "").trim();
}

// Get next prompt, supporting both text and voice each iteration.
async function getNextPrompt(defaultText) {
  if (VOICE_MODE) {
    const choice = await promptOnce(
      "🧑‍💻 Type a command, or enter 'v' for voice, or 'exit' to quit:\n> "
    );
    if (!choice) return defaultText;
    if (/^(exit|quit|q)$/i.test(choice)) return "__EXIT__";
    if (/^v$/i.test(choice)) {
      const heard = await listenForVoiceCommand();
      return heard || "";
    }
    return choice;
  } else {
    const typed = await promptOnce(
      "🧑‍💻 Type a command (or 'exit' to quit). Press Enter for default:\n> "
    );
    if (/^(exit|quit|q)$/i.test(typed)) return "__EXIT__";
    return typed || defaultText;
  }
}

/* --------------------------- Demo / Entry (REPL) -------------------------- */
async function init() {
  try {
    // First iteration can accept CLI args; after that we loop forever.
    const cliArg = process.argv.slice(2).join(" ").trim();
    let next = cliArg || (VOICE_MODE ? "" : "Create a weather app with HTML, CSS and JS that shows a 7-day forecast");

    // REPL loop
    for (;;) {
      if (!next) {
        next = await getNextPrompt("Create a todo app with HTML, CSS and JS");
      }
      if (next === "__EXIT__") {
        console.log("👋 Bye!");
        break;
      }
      if (!next.trim()) {
        console.log("ℹ️ Empty input. Skipping.");
      } else {
        console.log("🧾 Using prompt:", JSON.stringify(next));
        const final = await runWorkflow(next);
        console.log("\n🤖 Final:\n" + final + "\n");
      }
      // Ask again for the next round
      next = "";
    }
  } catch (err) {
    if (err?.status === 429) {
      console.error("⛔ Quota hit. Enable billing, wait for reset, or switch project/model.");
    } else {
      console.error("Error occurred:", err);
    }
  }
}

init();
