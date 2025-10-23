# LLM Fine-tuning Example

A simple example of fine-tuning GPT-2 using LoRA on Apple Silicon (M4 Pro).

## Setup
```bash
python3 -m venv llm_env
source llm_env/bin/activate
pip install torch transformers datasets accelerate peft
```

## Run
```bash
python simple_example.py
```

## Requirements
- Apple Silicon Mac (M1/M2/M3/M4)
- Python 3.11+
- 24GB RAM recommended
