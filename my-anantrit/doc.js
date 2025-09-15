const OPENAI_API_KEY = "apikey"

const client = new OpenAI({ apikey: OPENAI_API_KEY});

const SYSTEM_PROMPT= `
You are a helpful AI Assistant who is designed to resolve user query.
If you think, user query needs a tool invocation, just tell me the tool name with parameters.

Available tools:
- addTwoNumbers()
`

async function init(){
    const response = await client.chat.completions.create({
        model:'gpt-4.1-mini',
        messages: [{ role: 'user', content:'Hey there'}],
    });
    console.log(response.choices[0].message.content);
}

init();