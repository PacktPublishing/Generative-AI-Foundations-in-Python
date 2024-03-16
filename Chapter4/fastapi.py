from fastapi import FastAPI, HTTPException, Request
from langchain.llms import OpenAI
import os

# Initialize FastAPI app
app = FastAPI()

# Setup Langchain with GPT-3.5, alter the temperature and max_tokens for different results
llm = OpenAI(
    model_name="gpt-3.5-turbo-instruct",
    temperature=0.7,
    max_tokens=256,
    api_key=os.environ["OPENAI_API_KEY"],
)


@app.post("/generate/")
async def generate_text(request: Request):
    data = await request.json()
    prompt = data.get("prompt")
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")
    response = llm(prompt)
    return {"generated_text": response}
