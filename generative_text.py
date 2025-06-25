# STEP 1: INSTALL NECESSARY LIBRARIES
# Run the following in your terminal if not already installed:
# pip install transformers torch

# STEP 2: IMPORT REQUIRED LIBRARIES
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import textwrap
import os
import datetime

# STEP 3: INITIALIZE MODEL AND TOKENIZER
print("Loading GPT-2 model and tokenizer...")
model_name = "gpt2"  # You can also try 'gpt2-medium' or 'gpt2-large'

try:
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

model.eval()

# Use GPU if available, otherwise fallback to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Model loaded successfully on {device}")

# STEP 4: GET USER PROMPT
print("\nEnter a topic, sentence, or starting phrase for text generation:")
prompt = input("Your Prompt: ").strip()

# Use a default prompt if none is entered
if not prompt:
    prompt = "Artificial intelligence is changing the future"
    print(f"No input provided. Using default prompt: {prompt}")

# STEP 5: CONFIGURE GENERATION SETTINGS
print("\nEnter generation settings (press Enter to use default values):")
try:
    max_len = int(input("Max tokens to generate [Default 200]: ") or 200)
    temp = float(input("Temperature (creativity level) [Default 0.9]: ") or 0.9)
    top_k = int(input("Top-k sampling [Default 50]: ") or 50)
    top_p = float(input("Top-p nucleus sampling [Default 0.95]: ") or 0.95)
    num_return_sequences = int(input("Number of outputs to generate [Default 1]: ") or 1)
    save_to_file = input("Do you want to save the output to a file? (y/n): ").strip().lower() == 'y'
except Exception as e:
    print(f"Invalid input. Using default values. Reason: {e}")
    max_len, temp, top_k, top_p, num_return_sequences = 200, 0.9, 50, 0.95, 1
    save_to_file = False

# STEP 6: TOKENIZE USER INPUT
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

# STEP 7: GENERATE TEXT FROM GPT-2
print("\nGenerating text...")
try:
    outputs = model.generate(
        input_ids=input_ids,
        max_length=max_len,
        do_sample=True,
        temperature=temp,
        top_k=top_k,
        top_p=top_p,
        no_repeat_ngram_size=2,
        num_return_sequences=num_return_sequences,
        early_stopping=True
    )
except Exception as e:
    print(f"Error during generation: {e}")
    exit()

# STEP 8: DISPLAY RESULTS WITH FORMATTING
def format_paragraph(text, width=100):
    """Format generated text into readable paragraph blocks."""
    return '\n'.join(textwrap.wrap(text, width))

final_output = []

for idx, output in enumerate(outputs):
    generated = tokenizer.decode(output, skip_special_tokens=True)
    print(f"\nOutput {idx + 1}:\n{'-' * 80}")
    formatted = format_paragraph(generated)
    print(formatted)
    print('-' * 80)
    final_output.append(formatted)

# STEP 9: SAVE TO FILE (OPTIONAL)
if save_to_file:
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"textgen_output_{timestamp}.txt"
    try:
        with open(filename, "w", encoding="utf-8") as f:
            for i, paragraph in enumerate(final_output, 1):
                f.write(f"Output {i}:\n{paragraph}\n\n")
        print(f"\nOutput saved to file: {filename}")
    except Exception as e:
        print(f"Failed to save file: {e}")

# FINAL MESSAGE
print("\nText generation complete. You can rerun the script with a different prompt.")
