# STEP 1: INSTALL NECESSARY LIBRARIES
# Run the following in your terminal if not already installed:
# pip install transformers torch

# STEP 2: IMPORT REQUIRED LIBRARIES
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import textwrap
import datetime

# STEP 3: INITIALIZE MODEL AND TOKENIZER
print("Loading GPT-2 model and tokenizer...")
model_name = "gpt2"  # Options: 'gpt2', 'gpt2-medium', 'gpt2-large'

try:
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Ensure the pad token is set
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
print(f"Model loaded successfully on {device}")

# STEP 4: GET USER PROMPT
prompt = input("\nEnter your prompt: ").strip()
if not prompt:
    prompt = "Artificial intelligence is changing the future"
    print(f"Using default prompt: {prompt}")

# STEP 5: CONFIGURE GENERATION SETTINGS
try:
    max_len = int(input("Max tokens to generate [Default 200]: ") or 200)
    temp = float(input("Temperature (0–1) [Default 0.9]: ") or 0.9)
    top_k = int(input("Top-k sampling [Default 50]: ") or 50)
    top_p = float(input("Top-p sampling [Default 0.95]: ") or 0.95)
    num_return_sequences = int(input("Number of outputs [Default 1]: ") or 1)
    save_to_file = input("Save output to file? (y/n): ").strip().lower() == 'y'
except Exception as e:
    print(f"Invalid input, using defaults. Reason: {e}")
    max_len, temp, top_k, top_p, num_return_sequences = 200, 0.9, 50, 0.95, 1
    save_to_file = False

# STEP 6: TOKENIZE PROMPT
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# STEP 7: GENERATE TEXT
print("\nGenerating text...\n")
try:
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=max_len,
        do_sample=True,
        temperature=temp,
        top_k=top_k,
        top_p=top_p,
        num_return_sequences=num_return_sequences,
        pad_token_id=tokenizer.eos_token_id
    )
except Exception as e:
    print(f"Error during generation: {e}")
    exit()

# STEP 8: DISPLAY AND FORMAT OUTPUT
def format_paragraph(text, width=100):
    return '\n'.join(textwrap.wrap(text, width))

final_output = []

for idx, output in enumerate(outputs):
    decoded = tokenizer.decode(output, skip_special_tokens=True)
    formatted = format_paragraph(decoded)
    print(f"Output {idx + 1}:\n{'-' * 80}\n{formatted}\n{'-' * 80}")
    final_output.append(formatted)

# STEP 9: SAVE TO FILE
if save_to_file:
    filename = f"textgen_output_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(filename, "w", encoding="utf-8") as f:
        for i, text in enumerate(final_output, 1):
            f.write(f"Output {i}:\n{text}\n\n")
    print(f"\nOutput saved to file: {filename}")

print("\nText generation complete.")
