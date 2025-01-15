import argparse
import os
import json
import torch
import sacrebleu
import wandb
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, LlamaForCausalLM
from comet import download_model, load_from_checkpoint

# Add argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', type=str, required=True, help='Path to the model directory')
args = parser.parse_args()

# W&B Configuration
entity = "inaciovieira-alpha-crc"  # Your W&B username or organization
project = "llamafactory"           # Your W&B project name

# Retrieve the TIMESTAMP from environment variables
timestamp = os.environ.get("TIMESTAMP")
if not timestamp:
    raise RuntimeError("TIMESTAMP environment variable is not set!")

# Construct target run name using the timestamp
target_run_name = f"train_{timestamp}"

# Initialize W&B connection
api = wandb.Api()
runs = api.runs(f"{entity}/{project}", filters={"config.run_name": target_run_name})

if runs:
    target_run = runs[0]
    original_run_id = target_run.id
    print(f"Found run ID {original_run_id} for run name {target_run_name} for EVAL STAGE")
else:
    raise RuntimeError(f"No runs found with run_name: {target_run_name}")

wandb.init(
    project=project,
    id=original_run_id,
    resume="must"
)

# Directories and paths
base_dir = os.path.expanduser("~/LLaMA-Factory")
model_dir = args.model_dir  # From command line
test_dataset_path = os.path.join(base_dir, "data/BALS_de_test_dataset.json")
output_dir = os.path.join(base_dir, "evaluation/autoeval", os.path.basename(model_dir))
os.makedirs(output_dir, exist_ok=True)

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = LlamaForCausalLM.from_pretrained(
    model_dir,
    device_map="auto",
    torch_dtype=torch.bfloat16
)
model.eval()

# Load test data
with open(test_dataset_path, "r") as f:
    data = json.load(f)

def extract_translation(output_text):
    """Extract translation from model output."""
    try:
        # Print full output for debugging
        print("Raw output:", output_text)
        
        # Try to find JSON-like structure
        json_start = output_text.rfind("{")
        json_end = output_text.find("}", json_start) + 1
        if json_start != -1 and json_end != -1:
            json_str = output_text[json_start:json_end]
            print("Found JSON:", json_str)
            
            response = json.loads(json_str)
            translation = response.get("translation", "").strip()
            if translation:
                return translation
            else:
                print(f"No translation found in response: {response}")
        else:
            print("No JSON structure found in output")
            
            # Fallback: try to extract text after the input
            lines = output_text.split('\n')
            if len(lines) > 0:
                return lines[-1].strip()
                
    except Exception as e:
        print(f"Error extracting translation: {e}")
        print(f"Raw output: {output_text}")
    return ""

predictions = []
sources = []
references = []

for sample in data:
    system = sample["system"]
    instruction = sample["instruction"]
    input_text = sample["input"]
    reference = sample["output"]
    
    prompt = f"{system}\n{instruction}\n{input_text}"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate translation
    output_tokens = model.generate(**inputs, max_new_tokens=256, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    output_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    
    # Extract translation
    translation = extract_translation(output_text)
    
    if not translation:
        print(f"Warning: Empty translation for input: {input_text[:100]}...")
    
    predictions.append(translation)
    sources.append(input_text)
    references.append(reference)

    # Optional: print progress
    print(f"Processed {len(predictions)} samples", end="\r")

print("\nFinished processing all samples")

# Save translations
with open(os.path.join(output_dir, "translations.txt"), "w") as f:
    for pred in predictions:
        f.write(pred + "\n")

# Compute BLEU, TER, ChrF++
bleu = sacrebleu.corpus_bleu(predictions, [references])
ter = sacrebleu.corpus_ter(predictions, [references])
chrf = sacrebleu.corpus_chrf(predictions, [references])

with open(os.path.join(output_dir, "metrics.txt"), "w") as f:
    f.write(f"BLEU: {bleu.score:.2f}\n")
    f.write(f"TER: {ter.score:.2f}\n")
    f.write(f"ChrF++: {chrf.score:.2f}\n")

print(f"BLEU: {bleu.score:.2f}, TER: {ter.score:.2f}, ChrF++: {chrf.score:.2f}")

# COMET evaluation
print("Downloading COMET model...")
comet_model_path = download_model("Unbabel/wmt20-comet-da")
comet_model = load_from_checkpoint(comet_model_path)

comet_data = [{"src": s, "mt": t, "ref": r} for s, t, r in zip(sources, predictions, references)]
comet_scores = comet_model.predict(comet_data, batch_size=8, gpus=1)

# Convert COMET scores to float
comet_scores = [float(score) for score in comet_scores]
avg_comet = sum(comet_scores)/len(comet_scores) if comet_scores else 0.0

with open(os.path.join(output_dir, "comet_scores.txt"), "w") as f:
    for score in comet_scores:
        f.write(f"{score}\n")
    f.write(f"Average COMET score: {avg_comet:.4f}\n")

print(f"Average COMET score: {avg_comet:.4f}")

# Add W&B logging for metrics
wandb.log({
    "eval/bleu": bleu.score,
    "eval/ter": ter.score,
    "eval/chrf": chrf.score,
    "eval/comet": avg_comet
})

# Close W&B run
wandb.finish()


