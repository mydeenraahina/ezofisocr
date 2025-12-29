from transformers import AutoProcessor, AutoModelForVision2Seq
import torch

MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"

# Load processor and model (done only once per container)
processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForVision2Seq.from_pretrained(MODEL_NAME)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

def handler(event):
    """
    Receives: {"input": {"text_prompt": "your prompt here"}}
    Returns: {"output": "model reply"}
    """
    user_input = event.get("input", {})
    prompt = user_input.get("text_prompt", None)

    if not prompt:
        return {"output": "No prompt provided."}

    # Structure message for pure text input, as expected by Qwen VL chat template
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt}
            ]
        }
    ]

    # Prepare inputs and generate output
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

# For local test only
if __name__ == "__main__":
    test_event = {
        "input": {
            "text_prompt": "What animal is on the candy?"
        }
    }
    print(handler(test_event))
