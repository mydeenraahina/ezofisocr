import runpod
from transformers import AutoProcessor, AutoModelForVision2Seq
import torch

MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForVision2Seq.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)

def handler(event):
    prompt = event["input"].get("text_prompt")

    if not prompt:
        return {"error": "No prompt provided"}

    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt}]
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=64
    )

    result = processor.decode(
        outputs[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True
    )

    return {"output": result}

runpod.serverless.start({"handler": handler})
