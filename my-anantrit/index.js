// index.js
import { GoogleGenerativeAI, SchemaType } from "@google/generative-ai";
import { config } from "dotenv";
import { exec } from "node:child_process";
import { promises as fs } from "node:fs";
import path from "node:path";

const USE_SYNTH_FALLBACK = process.env.USE_SYNTH_FALLBACK === "1";

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

// For batch writes: normalize all file paths to live under baseDir
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

// For single writeFile: normalize path
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

/* --------------------------- Agent Loop ----------------------------------- */
async function runWorkflow(userQuestion, { maxSteps = 200 } = {}) {
  const model = client.getGenerativeModel({
    model: "gemini-1.5-flash",
    systemInstruction: SYSTEM_PROMPT,
  });

  const contents = [{ role: "user", parts: [{ text: userQuestion }] }];

  const baseDir = inferProjectDir(userQuestion);
  console.log("📂 Base project dir:", baseDir);

  // loop guards
  let lastActionSig = "";
  let sameActionCount = 0;
  const MAX_SAME_ACTION = 3;

  // do not allow "output" until a write actually succeeded
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

    // block premature output for creation tasks
    if (tag === "output") {
      const creationIntent = /create|build|generate|make/i.test(userQuestion);
      if (creationIntent && !didWriteAnything) {
        const observeObj = {
          step: "observe",
          tool: "",
          input: "",
          content:
            "You cannot emit an output yet—no files were written. " +
            "Now produce ONE action with tool=batchWriteFiles and a non-empty 'input' JSON. " +
            "Your file CONTENT must be tailored to this request and different from any examples.",
        };
        contents.push({
          role: "user",
          parts: [{ text: JSON.stringify(observeObj) }],
        });
        continue;
      }
      return step.content;
    }

    if (tag === "action") {
      const { tool, input } = parseToolandInput(step.tool, step.input);

      // one declaration only
      let fixedInput = input;

      // normalize model-supplied write paths into baseDir
      if (tool === "writeFile" && fixedInput && String(fixedInput).trim()) {
        try {
          fixedInput = rewriteSingleWriteToBaseDir(fixedInput, baseDir);
        } catch {}
      }
      if (
        tool === "batchWriteFiles" &&
        fixedInput &&
        String(fixedInput).trim()
      ) {
        try {
          fixedInput = rewriteBatchInputToBaseDir(fixedInput, baseDir);
        } catch {}
      }

      // loop-guard signature to detect repeated empty actions
      const sig = JSON.stringify({
        tool,
        hasInput: !!(fixedInput && String(fixedInput).trim()),
      });
      if (sig === lastActionSig) {
        sameActionCount++;
      } else {
        sameActionCount = 0;
        lastActionSig = sig;
      }

      // structured args → input JSON (writeFile)
      if (
        tool === "writeFile" &&
        !fixedInput &&
        step.args?.path &&
        step.args?.content
      ) {
        fixedInput = JSON.stringify({
          path: step.args.path,
          content: step.args.content,
        });
        try {
          fixedInput = rewriteSingleWriteToBaseDir(fixedInput, baseDir);
        } catch {}
      }

      // structured args → input JSON (batchWriteFiles)
      if (
        tool === "batchWriteFiles" &&
        !fixedInput &&
        Array.isArray(step.args?.files)
      ) {
        fixedInput = JSON.stringify({ files: step.args.files });
        try {
          fixedInput = rewriteBatchInputToBaseDir(fixedInput, baseDir);
        } catch {}
      }

      // If the model repeats an empty batchWriteFiles action, synthesize ONE write and finish
      if (
        tool === "batchWriteFiles" &&
        (!fixedInput || !String(fixedInput).trim())
      ) {
        if (sameActionCount >= MAX_SAME_ACTION) {
          if (!USE_SYNTH_FALLBACK) {
            // Nudge the model to emit a proper batchWriteFiles with real content
            const example = {
              step: "action",
              tool: "batchWriteFiles",
              input: JSON.stringify({
                files: [
                  {
                    path: `./${baseDir}/index.html`,
                    content: "<!doctype html>...</html>",
                  },
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
                "batchWriteFiles error: missing input. Provide JSON like " +
                '{"files":[{"path":"./file","content":"..."}]}.\n' +
                "Emit EXACTLY ONE action next with **your own** generated file contents (unique to this request), " +
                `and all paths under ./${baseDir}. Example shape:\n` +
                JSON.stringify(example),
            };

            contents.push({
              role: "user",
              parts: [{ text: JSON.stringify(observeObj) }],
            });
            continue;
          } else {
            // Optional safety scaffold (opt-in via USE_SYNTH_FALLBACK=1)
            const title = baseDir.replace(/-/g, " ");
            const synthesized = {
              files: [
                {
                  path: `./${baseDir}/index.html`,
                  content:
                    '<!doctype html><html lang="en"><head><meta charset="utf-8">' +
                    '<meta name="viewport" content="width=device-width,initial-scale=1">' +
                    `<title>${title}</title>` +
                    `<link rel="stylesheet" href="./styles.css">` +
                    `</head><body><div id="app"></div><script src="./script.js"></script></body></html>`,
                },
                {
                  path: `./${baseDir}/styles.css`,
                  content:
                    "html,body{margin:0;padding:0;font-family:system-ui}#app{max-width:960px;margin:0 auto;padding:2rem}\n",
                },
                {
                  path: `./${baseDir}/script.js`,
                  content: `document.getElementById('app').innerHTML = "<h1>${title}</h1>";\n`,
                },
              ],
            };

            const inputJson = JSON.stringify(synthesized);
            let observed;
            try {
              observed = await TOOLS.batchWriteFiles(inputJson);
              const folder = path.resolve(process.cwd(), baseDir);
              const files = await fs.readdir(folder).catch(() => []);
              console.log(`🔧 batchWriteFiles(synthesized) -> ${observed}`);
              console.log(`📁 WROTE FILES under ./${baseDir}:`, files);
              return `✅ Created scaffold in ${folder}:\n- ${files.join(
                "\n- "
              )}`;
            } catch (e) {
              console.error("❌ batchWriteFiles(synthesized) error:", e);
              contents.push({
                role: "user",
                parts: [
                  {
                    text: JSON.stringify({
                      step: "observe",
                      tool: "batchWriteFiles",
                      input: inputJson,
                      content: `Tool batchWriteFiles error: ${e?.message || e}`,
                    }),
                  },
                ],
              });
              continue;
            }
          }
        }
      }

      // validate tool
      if (!tool || !(tool in TOOLS)) {
        const observeObj = {
          step: "observe",
          tool,
          input: fixedInput,
          content: `Unsupported tool: ${tool || "(none)"}`,
        };
        contents.push({
          role: "user",
          parts: [{ text: JSON.stringify(observeObj) }],
        });
        continue;
      }

      // per-tool input rules
      if (tool === "readFile") {
        if (!fixedInput || !String(fixedInput).trim()) {
          const askedForPkg = /package\.json/i.test(userQuestion);
          if (askedForPkg) fixedInput = "./package.json";
        }
        if (!fixedInput || !String(fixedInput).trim()) {
          const observeObj = {
            step: "observe",
            tool,
            input: fixedInput,
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
        if (!fixedInput || !String(fixedInput).trim()) fixedInput = ".";
      }

      if (tool === "writeFile") {
        if (!fixedInput || !String(fixedInput).trim()) {
          const template = {
            step: "action",
            tool: "writeFile",
            input: JSON.stringify({
              path: `./${baseDir}/index.html`,
              content: "<!-- content omitted -->",
            }),
            content: "Create index.html",
          };
          const observeObj = {
            step: "observe",
            tool,
            input: fixedInput,
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
            input: fixedInput,
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
        // mark success on write tools
        if (
          (tool === "writeFile" || tool === "batchWriteFiles") &&
          typeof observed === "string" &&
          observed.startsWith("✅")
        ) {
          didWriteAnything = true;
          const files = await fs.readdir(`./${baseDir}`).catch(() => []);
          console.log(`📁 WROTE FILES under ./${baseDir}:`, files);
        }
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

  // If we got here without a model "output" but wrote something, summarize truthfully
  if (didWriteAnything) {
    const folder = path.resolve(process.cwd(), inferProjectDir(userQuestion));
    const files = await fs.readdir(folder).catch(() => []);
    return `✅ Files created in ${folder}:\n- ${files.join("\n- ")}`;
  }
  throw new Error("Max steps reached without OUTPUT");
}

/* --------------------------- Demo ----------------------------------------- */
async function init() {
  try {
    // Example request; change the text freely:
    const final = await runWorkflow(
      "Create a todo app with html, css and js. FUll working and with a beautiful UI"
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
