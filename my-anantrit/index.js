// index.js
import { GoogleGenerativeAI } from "@google/generative-ai";
import { config } from "dotenv";
import mic from "mic";
import fs from "fs";
import path from "path";

config();

const API_KEY = process.env.GEMINI_API_KEY;
if (!API_KEY) {
  console.error("❌ GEMINI_API_KEY missing in .env");
  process.exit(1);
}

const genAI = new GoogleGenerativeAI(API_KEY);

// ---- Recording helper -------------------------------------------------------
function recordOnce({ seconds = 5, file = "input.wav" } = {}) {
  return new Promise((resolve, reject) => {
    // Configure mic (16kHz mono WAV is a good default for speech)
    const micInstance = mic({
      rate: "16000",
      channels: "1",
      bitwidth: "16",
      fileType: "wav" // requires SoX on Windows
    });

    const micInputStream = micInstance.getAudioStream();
    const outPath = path.resolve(file);
    const writeStream = fs.createWriteStream(outPath);

    micInputStream.on("error", reject);
    writeStream.on("error", reject);
    writeStream.on("finish", () => resolve(outPath));

    micInputStream.pipe(writeStream);

    console.log(`🎙️  Recording... (speak now, ${seconds}s)`);
    micInstance.start();

    setTimeout(() => {
      micInstance.stop();
      console.log("⏹️  Stopped. Saving audio...");
    }, seconds * 1000);
  });
}

// ---- Call Gemini with audio -------------------------------------------------
async function askGeminiWithAudio(wavPath) {
  const audioBytes = fs.readFileSync(wavPath).toString("base64");
  const model = genAI.getGenerativeModel({ model: "gemini-1.5-flash" });

  // Prompt asks the model to first transcribe, then answer naturally
  const result = await model.generateContent([
    {
      text:
        "You will receive an audio clip. First transcribe the user's speech, " +
        "then respond helpfully in one or two sentences."
    },
    {
      inlineData: {
        data: audioBytes,
        mimeType: "audio/wav"
      }
    }
  ]);

  return result.response.text();
}

// ---- Main -------------------------------------------------------------------
async function init() {
  try {
    const wavPath = await recordOnce({ seconds: 5, file: "input.wav" });
    const reply = await askGeminiWithAudio(wavPath);
    console.log("\n🤖 Gemini:", reply);
  } catch (err) {
    console.error("Error occurred:", err);
  }
}

init();
