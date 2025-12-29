from transformers import AutoProcessor, AutoModelForVision2Seq
import torch

MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"

# Load processor and model (only once per pod/container)
processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForVision2Seq.from_pretrained(MODEL_NAME)
model = model.to("cuda" if torch.cuda.is_available() else "cpu")

def handler(event):
    """
    Expects event:
    {
      "input": {
        "messages": [
          {
            "role": "user",
            "content": [
              {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"},
              {"type": "text", "text": "What animal is on the candy?"}
            ]
          }
        ]
      }
    }
    Supports text-only by omitting the 'image' line.
    """
    user_input = event.get("input", {})
    messages = user_input.get("messages", [])
    if not messages:
        return {"output": "No messages provided."}

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

# Local debug/testing
if __name__ == "__main__":
    test_event = {
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"},
                        {"type": "text", "text": "What animal is on the candy?"}
                    ]
                }
            ]
        }
    }
    print(handler(test_event))
