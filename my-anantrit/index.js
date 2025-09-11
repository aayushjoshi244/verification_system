// index.js
// import mic from "mic";
// import fs from "fs";
// import path from "path";

// config();

// const API_KEY = process.env.GEMINI_API_KEY;
// if (!API_KEY) {
//   console.error("❌ GEMINI_API_KEY missing in .env");
//   process.exit(1);
// }

// const genAI = new GoogleGenerativeAI(API_KEY);

// // ---- Recording helper -------------------------------------------------------
// function recordOnce({ seconds = 5, file = "input.wav" } = {}) {
//   return new Promise((resolve, reject) => {
//     // Configure mic (16kHz mono WAV is a good default for speech)
//     const micInstance = mic({
//       rate: "16000",
//       channels: "1",
//       bitwidth: "16",
//       encoding: "signed-integer",
//       endian: "little",
//       fileType: "wav" // requires SoX on Windows
//     });

//     const micInputStream = micInstance.getAudioStream();
//     const outPath = path.resolve(file);
//     const writeStream = fs.createWriteStream(outPath);

//     micInputStream.on("error", reject);
//     writeStream.on("error", reject);
//     writeStream.on("finish", () => resolve(outPath));

//     micInputStream.pipe(writeStream);

//     console.log(`🎙️  Recording... (speak now, ${seconds}s)`);
//     micInstance.start();

//     setTimeout(() => {
//       micInstance.stop();
//       console.log("⏹️  Stopped. Saving audio...");
//     }, seconds * 1000);
//   });
// }

// function addTwoNumbers(X, Y) {
//   return X+Y;
// }

// const SYSTEM_PROMPT = `
//   You are Anantrit, a friendly and helpful AI assistant. You can understand and respond to voice commands. Always be polite and concise in your answers.
//   If you think, user query needs a tool invocation, just tell me the tool name with parameters.

//   Available tools:
//   - addTwoNumbers(X: number, Y: number)
// `;

// // ---- Call Gemini with audio -------------------------------------------------
// async function askGeminiWithAudio(wavPath) {
//   const audioBytes = fs.readFileSync(wavPath).toString("base64");

//   const model = genAI.getGenerativeModel({
//     model: "gemini-1.5-flash",
//     systemInstruction: SYSTEM_PROMPT
//    });

//   // Prompt asks the model to first transcribe, then answer naturally
//   const result = await model.generateContent([
//     {
//       text:
//         "Transscribe the audio and then answer helpfully."
//     },
//     {
//       inlineData: {
//         data: audioBytes,
//         mimeType: "audio/wav"
//       }
//     }
//   ]);

//   return result.response.text();
// }

// // ---- Main -------------------------------------------------------------------
// async function init() {
//   try {
//     const wavPath = await recordOnce({ seconds: 5, file: "input.wav" });
//     const reply = await askGeminiWithAudio(wavPath);
//     console.log("\n🤖 ANANTRIT:", reply);
//   } catch (err) {
//     console.error("Error occurred:", err);
//   }
// }

// init();

// index.js
// index.js
import { GoogleGenerativeAI, SchemaType } from "@google/generative-ai";
import { config } from "dotenv";
import { exec } from "node:child_process";
import { promises as fs } from "node:fs";
import path from "node:path";

config();

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
  if (!relPath || !content) {
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
  batchWriteFiles: batchWriteFilesTool, // ⬅️ add this
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
- For a small website (index.html, styles.css, script.js), use a single "batchWriteFiles" action with "files": [{path,content},...].


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
- To CREATE or MODIFY files, always use "writeFile". Do NOT use shell redirection (echo >>, etc).
- For "writeFile", put a JSON string in "input", e.g.: {"path":"./index.html","content":"<html>...</html>"}.
- To CREATE or MODIFY files, always use "writeFile".
- For "writeFile", you may provide either:
  (a) "input" as a JSON string: {"path":"./index.html","content":"..."} OR
  (b) "args" as an object: {"path":"./index.html","content":"..."}.
- Do not emit a writeFile action without including path and content.


Examples:

START: What is in my package.json?
THINK: The user wants the contents of package.json in CWD. I should read "./package.json" using readFile.
ACTION: {"step":"action","tool":"readFile","input":"./package.json","content":"Reading package.json"}
OBSERVE: {"step":"observe","tool":"readFile","input":"./package.json","content":"<file text here>"}
THINK: I will summarize name, version, and scripts, then present the content (truncated if long).
OUTPUT: {"step":"output","content":"name: X, version: Y, scripts: {...}\n\n<file text or truncated excerpt>"}

START: Show files here
THINK: I should list the current directory using listDir with default path "."
ACTION: {"step":"action","tool":"listDir","input":".","content":"Listing current directory"}
OBSERVE: {"step":"observe","tool":"listDir","input":".","content":"[{\"name\":\"package.json\",\"type\":\"file\"}, {\"name\":\"src\",\"type\":\"dir\"}]"}
THINK: I can now describe the directory contents.
OUTPUT: {"step":"output","content":"Found 2 items: package.json (file), src (dir)."}
`;

/* --------------------------- JSON Schema ---------------------------------- */
const generationConfig = {
  responseMimeType: "application/json",
  responseSchema: {
    type: SchemaType.OBJECT,
    properties: {
      step: { type: SchemaType.STRING, enum: ["think", "action", "observe", "output"] },
      tool: {
        type: SchemaType.STRING,
        enum: ["readFile","listDir","pwd","executeCommand","writeFile","batchWriteFiles"], // add it here
        nullable: true,
      },
      input: { type: SchemaType.STRING, nullable: true },
      args: {
        type: SchemaType.OBJECT,
        nullable: true,
        properties: {
          path: { type: SchemaType.STRING, nullable: true },
          content: { type: SchemaType.STRING, nullable: true },
          // allow structured batching
          files: {
            type: SchemaType.ARRAY,
            nullable: true,
            items: {
              type: SchemaType.OBJECT,
              properties: {
                path: { type: SchemaType.STRING },
                content: { type: SchemaType.STRING }
              }
            }
          }
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

/* --------------------------- Agent Loop ----------------------------------- */
async function runWorkflow(userQuestion, { maxSteps = 200 } = {}) {
  const model = client.getGenerativeModel({
    model: "gemini-1.5-flash",
    systemInstruction: SYSTEM_PROMPT,
  });

  const contents = [{ role: "user", parts: [{ text: userQuestion }] }];
  let lastActionSig = "";
  let sameActionCount = 0;
  const MAX_SAME_ACTION = 2;


  for (let i = 0; i < maxSteps; i++) {
    const res = await generateWithRetry(model, { contents, generationConfig });
    const text = res.response.text();
    const step = safeParseJSON(text);

    if (!step) {
      // Ask for a corrected JSON step instead of crashing
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
      return step.content;
    }

    if (tag === "action") {
      const { tool, input } = parseToolandInput(step.tool, step.input);

      // one declaration only
      let fixedInput = input;

      // loop guard signature (did we just repeat the same empty action?)
      const sig = JSON.stringify({
        tool,
        hasInput: !!(fixedInput && String(fixedInput).trim()),
        content:
          typeof step.content === "string" ? step.content.slice(0, 60) : "",
      });
      if (sig === lastActionSig) {
        sameActionCount++;
      } else {
        sameActionCount = 0;
        lastActionSig = sig;
      }

      if (sameActionCount >= MAX_SAME_ACTION) {
        // Force a single batchWriteFiles template so the agent stops looping
        const example = {
          step: "action",
          tool: "batchWriteFiles",
          input: JSON.stringify({
            files: [
              {
                path: "./index.html",
                content:
                  '<!doctype html><html><head><meta charset="utf-8"><title>Todo</title>' +
                  '<link rel="stylesheet" href="./styles.css"></head>' +
                  '<body><div id="app"></div><script src="./script.js"></script></body></html>',
              },
              {
                path: "./styles.css",
                content:
                  "body{font-family:system-ui;margin:0;padding:2rem;background:#f6f7f9}#app{max-width:700px;margin:0 auto}ul{list-style:none;padding:0}li{display:flex;align-items:center;gap:.5rem;padding:.5rem;background:#fff;margin:.5rem 0;border-radius:8px;box-shadow:0 1px 2px rgba(0,0,0,.06)}input[type=text]{padding:.5rem;border:1px solid #ccc;border-radius:6px}button{padding:.5rem .75rem;border:0;border-radius:6px;cursor:pointer}",
              },
              {
                path: "./script.js",
                content:
                  "const app=document.getElementById('app');const input=document.createElement('input');input.type='text';input.placeholder='New todo';const add=document.createElement('button');add.textContent='Add';const list=document.createElement('ul');app.append(input,add,list);const todos=[];function render(){list.innerHTML='';todos.forEach((t,i)=>{const li=document.createElement('li');const cb=document.createElement('input');cb.type='checkbox';cb.checked=t.done;cb.onchange=()=>{t.done=!t.done;render()};const span=document.createElement('span');span.textContent=t.text;span.style.textDecoration=t.done?'line-through':'none';const del=document.createElement('button');del.textContent='×';del.onclick=()=>{todos.splice(i,1);render()};li.append(cb,span,del);list.append(li);});}add.onclick=()=>{const v=input.value.trim();if(!v)return;todos.push({text:v,done:false});input.value='';render();};render();",
              },
            ],
          }),
          content: "Create todo app files",
        };
        const observeObj = {
          step: "observe",
          tool,
          input: fixedInput,
          content:
            "You are repeating the same action without progress. Use a single batchWriteFiles to create all files.\n" +
            "Emit EXACTLY this next step (you may adjust file contents if desired):\n" +
            JSON.stringify(example),
        };
        contents.push({
          role: "user",
          parts: [{ text: JSON.stringify(observeObj) }],
        });
        continue;
      }

      // convert structured args -> input JSON for writeFile
      if (
        tool === "writeFile" &&
        !fixedInput &&
        step.args &&
        step.args.path &&
        step.args.content
      ) {
        fixedInput = JSON.stringify({
          path: step.args.path,
          content: step.args.content,
        });
      }

      // convert structured args -> input for batchWriteFiles
      if (
        tool === "batchWriteFiles" &&
        !fixedInput &&
        step.args &&
        Array.isArray(step.args.files)
      ) {
        fixedInput = JSON.stringify({ files: step.args.files });
      }

      if (tool === "batchWriteFiles") {
        if (!fixedInput || !String(fixedInput).trim()) {
          const example = {
            step: "action",
            tool: "batchWriteFiles",
            input: JSON.stringify({
              files: [
                { path: "./index.html", content: "<!doctype html>..." },
                { path: "./styles.css", content: "/* css */" },
                { path: "./script.js", content: "// js" },
              ],
            }),
            content: "Create website files",
          };
          const observeObj = {
            step: "observe",
            tool,
            input,
            content:
              'batchWriteFiles error: missing input. Provide JSON like {"files":[{"path":"./file","content":"..."}]}.\n' +
              "Emit EXACTLY like this, updating file contents as needed:\n" +
              JSON.stringify(example),
          };
          contents.push({
            role: "user",
            parts: [{ text: JSON.stringify(observeObj) }],
          });
          continue;
        }
      }

      // validate tool early
      if (!tool || !(tool in TOOLS)) {
        const observeObj = {
          step: "observe",
          tool,
          input,
          content: `Unsupported tool: ${tool || "(none)"}`,
        };
        contents.push({
          role: "user",
          parts: [{ text: JSON.stringify(observeObj) }],
        });
        continue;
      }

      // ---- per-tool input rules ----
      if (tool === "readFile") {
        if (!fixedInput || !String(fixedInput).trim()) {
          const askedForPkg = /package\.json/i.test(userQuestion);
          if (askedForPkg) fixedInput = "./package.json";
        }
        if (!fixedInput || !String(fixedInput).trim()) {
          const observeObj = {
            step: "observe",
            tool,
            input,
            content: `readFile error: missing path. Example: "./package.json".`,
          };
          contents.push({
            role: "user",
            parts: [{ text: JSON.stringify(observeObj) }],
          });
          continue;
        }
      }

      if (tool === "listDir") {
        // optional input; defaults to "."
        if (!fixedInput || !String(fixedInput).trim()) fixedInput = ".";
      }

      if (tool === "writeFile") {
        // if missing, return a copy-pasteable template
        if (!fixedInput || !String(fixedInput).trim()) {
          const template = {
            step: "action",
            tool: "writeFile",
            input: JSON.stringify({
              path: "./index.html",
              content:
                '<!doctype html><html><head><meta charset="utf-8"><title>Todo</title>' +
                '<link rel="stylesheet" href="./styles.css"></head>' +
                '<body><div id="app"></div><script src="./script.js"></script></body></html>',
            }),
            content: "Create index.html",
          };

          const observeObj = {
            step: "observe",
            tool,
            input,
            content:
              'writeFile error: missing input. Provide JSON string like {"path":"./file","content":"..."}. ' +
              "Emit your next step EXACTLY like this example, replacing content as needed:\n" +
              JSON.stringify(template),
          };
          contents.push({
            role: "user",
            parts: [{ text: JSON.stringify(observeObj) }],
          });
          continue;
        }
      }

      if (tool === "executeCommand") {
        // fallback: treat content as command if it looks like one
        if (
          (!fixedInput || !String(fixedInput).trim()) &&
          typeof step.content === "string"
        ) {
          const looksLikeCmd =
            /^[\s]*(echo|printf|type|cat|powershell|cmd|npm|npx|mkdir|dir|ls|touch)\b/i.test(
              step.content
            );
          if (looksLikeCmd) fixedInput = step.content.trim();
        }
        if (!fixedInput || !String(fixedInput).trim()) {
          const observeObj = {
            step: "observe",
            tool,
            input,
            content: `Missing input for executeCommand. Provide a full command string.`,
          };
          contents.push({
            role: "user",
            parts: [{ text: JSON.stringify(observeObj) }],
          });
          continue;
        }
      }

      // run tool
      let observed;
      try {
        observed = await TOOLS[tool](fixedInput);
      } catch (e) {
        observed = `Tool ${tool} error: ${e?.message || e}`;
      }

      console.log(
        `🔧 ${tool}(${fixedInput ?? ""}) -> ${String(observed).slice(0, 200)}${
          String(observed).length > 200 ? "…" : ""
        }`
      );
      const observeObj = {
        step: "observe",
        tool,
        input: fixedInput,
        content: String(observed),
      };
      contents.push({
        role: "user",
        parts: [{ text: JSON.stringify(observeObj) }],
      });
      continue;
    }

    // THINK (or unknown tag): nudge to proceed
    contents.push({
      role: "user",
      parts: [
        { text: `{"step":"continue","content":"Proceed to the next step."}` },
      ],
    });
  }

  throw new Error("Max steps reached without OUTPUT");
}

/* --------------------------- Demo ----------------------------------------- */
async function init() {
  try {
    // Ask something agentic:
    const final = await runWorkflow(
      "Create a todo app with html and css, with working UI"
    );
    console.log("\n🤖 Final:\n" + final);
  } catch (err) {
    if (err?.status === 429) {
      console.error(
        "⛔ Quota hit. Enable billing, wait for reset, or switch project/model."
      );
    } else {
      console.error("Error occurred:", err);
    }
  }
}

init();
