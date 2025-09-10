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


// import OpenAI from 'openai';
import { GoogleGenerativeAI } from "@google/generative-ai";
import { config } from "dotenv";

config();

const GEMINI_API_KEY = process.env.GEMINI_API_KEY;

const client = new GoogleGenerativeAI(GEMINI_API_KEY);

const SYSTEM_PROMPT = `
  You are Anantrit, a friendly and helful AI assistant who is designed to resolve user query
  You work on START, THINK, ACTION, OBSERVE and OUTPUT Mode.

  In the start phase, user gives a query to you.
  Then, you THINK how to resolve the query atleast 3-4 times and make sure that all is clear
  If there is a need to call a tool, you call and ACTION event with tool and input parameters.
  If there is an action call, wait for the OBSERVE that is the output of the tool
  Based on the OBSERVE from previous step, you either output or repeat the loop

  Rules:
  - Always wait for next step
  - Always output a single step and wait for the next step
  - Output must be strictly JSON
  - Only call tool action from Available tools only.
  - Stricly follow the Output Format in JSON

  Available tools:
  - getWeatherInfo(city: string): string 

  Example:
  START: What is weather of Patiala?
  THINK: The user is asking for the weather of Patiala.
  THINK: From the available tools, I must call getWeatherInfo for Patiala as input
  ACTION: Call Tool getWeatherInfo(Patiala)
  OBSERVE: 32 Degree C
  THINK: The output of getWeatherInfo for Patiala is 32 Degree C
  OUTPUT: Hey, The weather of Patiala is 32 Degree C which is quite hot🥵

  Output Example:
  {"role": "user", "content":"What is weather of Patiala?"}
  {"step": "think", "content":"The user is asking for the weather of Patiala."}
  {"step": "think", "content":"From the available tools, I must call getWeatherInfo for Patiala as input"}
  {"step": "action", "tool":"getWeatherInfo", "input":"Patiala"}
  {"step": "observe", "content":"32 Degree C"}
  {"step": "think", "content":"The output of getWeatherInfo for Patiala is 32 Degree C"}
  {"step": "output", "content":"Hey, The weather of Patiala is 32 Degree C which is quite hot🥵"}

  Output Format:
  {"step": "string", "tool": "string", "input": "string", "content": "string"}
`
const messages = [
  role: "system", content: SYSTEM_PROMPT
]
const userQuery = ''

while (true){
  const response = client.getGenerativeModel({
      model: 'gemini-1.5-flash',
      systemInstruction: SYSTEM_PROMPT,
      response_format: {type: 'json_object'},
      messages: messages
    });

    messages.push({role: "assistant", 'content': response.choices[0].message.content })
    const parsed_response = JSON.parse(response.choices[0].message.content )

    if (parsed_response.step && parsed_response.step == "think") {
      console.log(`🧠: ${parsed_response.content}`);
      continue;
    }
}

async function init() {
  try {
    const model = client.getGenerativeModel({
      model: 'gemini-1.5-flash',
      systemInstruction: SYSTEM_PROMPT
    });

    const genConfig = {
      responseMimeType: "application/json",
      responseSchema: {
        type: "OBJECT",
        properties:{
          step: { type: "STRING" },
          tool: { type: "STRING", nullable: true },
          input: { type: "STRING", nullable: true },
          content: { type: "STRING" }
        },
        required: ["step", "content"]
      }
    }

    const response = await model.generateContent({
      contents: [
        { role: "user", parts: [{ text: "what is weather of Delhi?" }] },
        { role: "assistant", parts: [{ text: '{"step": "think", "content": "The user is asking for the weather of Delhi."}' }] },
        { role: "assistant", parts: [{ text: '{"step": "think", "content": "From the available tools, I must call getWeatherInfo for Delhi as input"}' }] },
        { role: "assistant", parts: [{ text: '{"step": "action", "content": "Calling the getWeatherInfo tool.", "tool": "getWeatherInfo(Delhi)"}' }] },
        { role: "assistant", parts: [{ text: '{"step": "observe", "content": "42 Degree C""}' }] },
        { role: "assistant", parts: [{ text: '{"step": "think", "content": "The output of getWeatherInfo for Delhi is 42 Degree C. I should formulate a user-friendly response."}' }] },

      ],
      generationConfig: genConfig
    });

    const text = response.response.text();
    console.log("Anantrit:", text);
  } catch (err) {
    // Quota / rate-limit handling
    if (err?.status === 429) {
      const details = JSON.stringify(err.errorDetails || []);
      if (details.includes("GenerateRequestsPerDayPerProjectPerModel-FreeTier")) {
        console.error("⛔ Free-tier daily quota exceeded for this project/model.");
        console.error("👉 Options: wait for reset, enable billing, or switch project/model.");
      } else {
        // Some 429s include a retryDelay (burst rate-limits)
        const m = details.match(/"retryDelay":"(\d+)s"/);
        if (m) console.error(`⏳ Rate limited. Try again after ~${m[1]}s.`);
        else console.error("⏳ Rate limited. Try again shortly.");
      }
    } else {
      console.error("Error occurred:", err);
    }
  }
}    

init();