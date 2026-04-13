# Travel planner — credentials and environment

Do not commit real tokens. Put secrets only in **`.env`** (gitignored).

## Required

| Variable | Where |
|----------|--------|
| **`HF_TOKEN`** | [Hugging Face tokens](https://huggingface.co/settings/tokens) (read access is enough) |

## Optional

| Variable | Default / notes |
|----------|-----------------|
| **`HF_MODEL_ID`** | Example: `mistralai/Mistral-7B-Instruct-v0.3`. If unset, the app tries Mistral v0.3 → v0.2 → Qwen2.5-7B → Phi-3-mini. If set, that model is tried first, then the rest. |
| **`HF_MAX_TOKENS`** | `1200` |
| **`HF_TEMPERATURE`** | `0.7` |
| **`HF_INFERENCE_PROVIDER`** | If **`.env` exists** in this folder, only values **in that file** apply. Unset there → `auto` for `hf_` tokens, else `hf-inference`. |

## Setup

Copy **`.env.example`** → **`.env`**, set **`HF_TOKEN`**.

If a token was exposed, revoke it on Hugging Face and create a new one in **`.env`** only.

Enable inference providers if you see “model not supported”: [Inference providers](https://huggingface.co/settings/inference-providers).
