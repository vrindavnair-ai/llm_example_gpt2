"""
Simple Fine-tuning Example - Step by Step:
Step 1: Install Dependencies
bash# Create a new environment (recommended)
python3 -m venv llm-finetune
source llm-finetune/bin/activate

# Install required packages
pip install torch transformers datasets accelerate peft
"""

import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model
from datasets import Dataset

# Check if MPS is available
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# 1) Load a small model (GPT-2 small - only 124M parameters)
model_name = "gpt2"
print(f"Loading model: {model_name}")

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16
)

# 2) Configure LoRA for efficient fine-tuning
print("Setting up LoRA...")
lora_config = LoraConfig(
    r=8,  # Low rank
    lora_alpha=16,
    target_modules=["c_attn"],  # GPT-2 attention layers
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 3) Create a simple dataset (teaching the model about AI)
training_texts = [
    "Artificial Intelligence is the simulation of human intelligence by machines.",
    "Machine Learning is a subset of AI that learns from data.",
    "Deep Learning uses neural networks with multiple layers.",
    "Natural Language Processing helps computers understand human language.",
    "Computer Vision enables machines to interpret visual information.",
    "AI can be used for image recognition, speech recognition, and text generation.",
    "Neural networks are inspired by the human brain structure.",
    "Training AI models requires large amounts of data and computation.",
]

# Create dataset
dataset_dict = {"text": training_texts}
dataset = Dataset.from_dict(dataset_dict)

# 4) Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(
        examples["text"], 
        truncation=True, 
        padding="max_length",
        max_length=128
    )

tokenized_dataset = dataset.map(
    tokenize_function, 
    batched=True,
    remove_columns=["text"]
)

# 5) Set up training
print("Starting training...")
training_args = TrainingArguments(
    output_dir="./gpt2-finetuned",
    num_train_epochs=10,
    per_device_train_batch_size=2,
    learning_rate=2e-4,
    logging_steps=5,
    save_strategy="epoch",
    report_to="none",  # Disable wandb
    use_mps_device=True if device == "mps" else False
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # We're doing causal language modeling, not masked
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)

# 6) Train!
trainer.train()

# 7) Save the model
print("Saving model...")
model.save_pretrained("./gpt2-finetuned-final")
tokenizer.save_pretrained("./gpt2-finetuned-final")

print("Training complete! Model saved to ./gpt2-finetuned-final")