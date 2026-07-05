"""Minimal OpenAI-compatible chat endpoint for ITERRET's LLM calls.
Backed by plain Qwen3-4B-Instruct (no delta-mem adapter).
"""
import time
import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "/home/kbasu/arnavbhatt/workmem_test/models/Qwen3-4B-Instruct-2507"
MAX_INPUT_TOKENS = 3072
MAX_OUTPUT_TOKENS = 512

app = FastAPI()

print("[mini_llm_server] Loading tokenizer...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

print("[mini_llm_server] Loading model (bfloat16, eager)...", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",
    local_files_only=True,
)
model.eval()
print("[mini_llm_server] Model ready.", flush=True)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: float = 0.0
    max_tokens: int = MAX_OUTPUT_TOKENS


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/v1/models")
def list_models():
    return {"data": [{"id": "Qwen/Qwen3-4B-Instruct-2507", "object": "model"}]}


@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest):
    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    encoded = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True,
    )
    input_ids = encoded["input_ids"]
    if input_ids.shape[1] > MAX_INPUT_TOKENS:
        input_ids = input_ids[:, -MAX_INPUT_TOKENS:]
    input_ids = input_ids.to("cuda:0")
    attention_mask = torch.ones_like(input_ids)
    max_new = min(int(req.max_tokens), MAX_OUTPUT_TOKENS)
    with torch.inference_mode():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new,
            do_sample=(req.temperature > 0.01),
            temperature=max(float(req.temperature), 0.01),
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
    del out, input_ids, attention_mask
    torch.cuda.empty_cache()
    return {
        "id": "chatcmpl-local",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
