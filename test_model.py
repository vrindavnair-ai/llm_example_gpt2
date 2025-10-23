from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load the fine-tuned model
base_model = AutoModelForCausalLM.from_pretrained("gpt2")
model = PeftModel.from_pretrained(base_model, "./gpt2-finetuned-final")
tokenizer = AutoTokenizer.from_pretrained("./gpt2-finetuned-final")

# Generate text
prompt = "Artificial Intelligence is"
inputs = tokenizer(prompt, return_tensors="pt")

outputs = model.generate(
    **inputs,
    max_length=50,
    num_return_sequences=1,
    temperature=0.7,
    do_sample=True
)

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Prompt: {prompt}")
print(f"Generated: {generated_text}")