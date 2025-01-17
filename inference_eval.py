import argparse
import os
import json
import sacrebleu
import wandb
from comet import download_model, load_from_checkpoint

# Add argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', type=str, required=True, help='Path to the model directory')
args = parser.parse_args()

# W&B Configuration
entity = "inaciovieira-alpha-crc"
project = "llamafactory"

# Retrieve the TIMESTAMP from environment variables
timestamp = os.environ.get("TIMESTAMP")
if not timestamp:
    raise RuntimeError("TIMESTAMP environment variable is not set!")

# Initialize W&B connection
api = wandb.Api()
runs = api.runs(f"{entity}/{project}", filters={"config.run_name": f"train_{timestamp}"})

if runs:
    target_run = runs[0]
    original_run_id = target_run.id
    print(f"Found run ID {original_run_id} for run name train_{timestamp} for EVAL STAGE")
else:
    raise RuntimeError(f"No runs found with run_name: train_{timestamp}")

wandb.init(
    project=project,
    id=original_run_id,
    resume="must"
)

# Directories and paths
base_dir = os.path.expanduser("~/LLaMA-Factory")
model_dir = args.model_dir
output_dir = os.path.join(base_dir, "evaluation/autoeval", os.path.basename(model_dir))
os.makedirs(output_dir, exist_ok=True)

def load_vllm_predictions(predictions_file):
    if not os.path.exists(predictions_file):
        raise FileNotFoundError(f"Predictions file not found: {predictions_file}")
        
    predictions = []
    sources = []
    references = []
    
    with open(predictions_file, 'r') as f:
        for i, line in enumerate(f, 1):
            try:
                data = json.loads(line)
                prompt = data['prompt']
                input_text = prompt.split('\n')[-1].strip('"')
                
                pred_json = json.loads(data['predict'])
                ref_json = json.loads(data['label'])
                
                if 'translation' not in pred_json or 'translation' not in ref_json:
                    print(f"Warning: Missing translation in line {i}")
                    continue
                    
                predictions.append(pred_json['translation'])
                sources.append(input_text)
                references.append(ref_json['translation'])
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Error processing line {i}: {e}")
                continue
            
    if not predictions:
        raise ValueError("No valid predictions found in file")
            
    return predictions, sources, references

# Load predictions from vLLM output
predictions_file = os.path.join(model_dir, f"predictions_{timestamp}.json")
if not os.path.exists(predictions_file):
    raise FileNotFoundError(f"Predictions file not found: {predictions_file}")

predictions, sources, references = load_vllm_predictions(predictions_file)
print(f"Loaded {len(predictions)} predictions")

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


