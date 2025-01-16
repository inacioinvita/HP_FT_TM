import argparse
import os
import json
import torch
import sacrebleu
import wandb
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, LlamaForCausalLM
from comet import download_model, load_from_checkpoint
from transformers import StoppingCriteria, StoppingCriteriaList


# Custom stopping criterion
class StopSequenceCriteria(StoppingCriteria):
    def __init__(self, stop_string, tokenizer):
        super().__init__()
        self.stop_string = stop_string
        self.tokenizer = tokenizer

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> bool:
        decoded_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        return self.stop_string in decoded_text
    
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


import re
import json

def extract_translation(output_text):
    """Extract the *first* valid JSON block containing 'translation' from the model output."""
    print("Raw output:", output_text)  # for debugging

    # Regex pattern to capture non-nested {...} blocks.
    # This is a simplistic pattern; if you have nested JSON braces, it won't handle those fully.
    pattern = r"\{[^{}]*\}"

    # Find all candidate substrings enclosed by braces
    candidates = re.findall(pattern, output_text)

    # Check each candidate in order until we find valid JSON with "translation" key
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if "translation" in parsed:
                translation = parsed["translation"].strip()
                if translation:
                    print("Found valid JSON:", candidate)
                    return translation
        except json.JSONDecodeError:
            # Not valid JSON; skip
            pass

    # If we reach here, no valid JSON with 'translation' was found
    print("No valid JSON with 'translation' found. Falling back to last line.")
    lines = output_text.split('\n')
    return lines[-1].strip() if lines else ""


predictions = []
sources = []
references = []
number_samples = 100
data = data[:number_samples]
# Define our stopping criteria
stop_criteria = StoppingCriteriaList([StopSequenceCriteria('}\n', tokenizer)])

for sample in data:
    system = sample["system"]
    instruction = sample["instruction"]
    input_text = sample["input"]
    reference = sample["output"]
    
    prompt = f"{system}\n{instruction}\n{input_text}"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate translation
    output_tokens = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.2,            # Might help reduce repeated patterns
        stopping_criteria=stop_criteria
    )
 
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

print(f"Total samples in dataset: {len(data)}")
print(f"Processing first {number_samples} samples")

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


