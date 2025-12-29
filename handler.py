from transformers import AutoProcessor, AutoModelForVision2Seq
import torch
from fastapi import FastAPI, Request
import uvicorn

MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"

# Load processor and model (only once at startup)
processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForVision2Seq.from_pretrained(MODEL_NAME)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

app = FastAPI()

@app.post("/infer")
async def infer(request: Request):
    data = await request.json()
    user_input = data.get("input", {})
    prompt = user_input.get("text_prompt", None)

    if not prompt:
        return {"output": "No prompt provided."}

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt}
            ]
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    outputs = model.generate(**inputs, max_new_tokens=40)
    generated_text = processor.decode(outputs[0][inputs["input_ids"].shape[-1]:])
    return {"output": generated_text}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
