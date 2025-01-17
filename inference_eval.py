import json
import sacrebleu
from comet import download_model, load_from_checkpoint
import torch
import os

def load_vllm_predictions(predictions_file):
    predictions = []
    sources = []
    references = []
    error_samples = []
    
    with open(predictions_file, 'r') as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
                
            try:
                data = json.loads(line)
                # Extract source from prompt
                prompt = data['prompt']
                user_text = prompt.split('user<|end_header_id|>\n\n')[1]
                source = user_text.split('<|eot_id|>')[0].strip('"')
                
                # Parse prediction and reference
                try:
                    # Handle case where predict is already a dict
                    if isinstance(data['predict'], dict):
                        pred_json = data['predict']
                    else:
                        # Clean and parse predict string
                        pred_str = data['predict'].strip()
                        if pred_str.startswith('"') and pred_str.endswith('"'):
                            pred_str = pred_str[1:-1]  # Remove outer quotes
                        pred_json = json.loads(pred_str)
                except Exception as e:
                    print(f"\nError parsing prediction at line {i}:")
                    print(f"Raw prediction: {data['predict']}")
                    raise e
                
                try:
                    # Clean and parse reference
                    ref_str = data['label'].replace('<|eot_id|>', '').strip()
                    if ref_str.startswith('"') and ref_str.endswith('"'):
                        ref_str = ref_str[1:-1]  # Remove outer quotes
                    ref_json = json.loads(ref_str)
                except Exception as e:
                    print(f"\nError parsing reference at line {i}:")
                    print(f"Raw reference: {data['label']}")
                    raise e
                
                predictions.append(pred_json['translation'])
                sources.append(source)
                references.append(ref_json['translation'])
                
            except Exception as e:
                error_samples.append({
                    'line': i,
                    'error': str(e),
                    'raw_data': line.strip()[:200]  # First 200 chars
                })
                continue
    
    if not predictions:
        raise ValueError("No valid predictions found in file")
    
    print(f"\nSuccessfully loaded {len(predictions)} prediction triplets")
    print(f"Found {len(error_samples)} errors")
    
    # Print first 5 error samples for analysis
    if error_samples:
        print("\nSample errors (first 5):")
        for sample in error_samples[:5]:
            print(f"\nLine {sample['line']}:")
            print(f"Error: {sample['error']}")
            print(f"Data: {sample['error']}")
    
    return predictions, sources, references

def compute_metrics(predictions, sources, references):
    # BLEU
    bleu = sacrebleu.metrics.BLEU()
    bleu_score = bleu.corpus_score(predictions, [references])
    
    # chrF
    chrf = sacrebleu.metrics.CHRF()
    chrf_score = chrf.corpus_score(predictions, [references])
    
    # TER
    ter = sacrebleu.metrics.TER()
    ter_score = ter.corpus_score(predictions, [references])
    
    # COMET
    print("Downloading COMET model...")
    model_path = download_model("Unbabel/wmt22-comet-da")
    print(f"Loading COMET model from {model_path}")
    comet_model = load_from_checkpoint(model_path)
    
    print("Computing COMET scores...")
    comet_data = [{"src": src, "mt": pred, "ref": ref} 
                  for src, pred, ref in zip(sources, predictions, references)]
    comet_score = comet_model.predict(comet_data, batch_size=32, gpus=1)
    
    return {
        "bleu": bleu_score.score,
        "chrf": chrf_score.score,
        "ter": ter_score.score,
        "comet": float(comet_score.system_score)
    }

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions_file", type=str, required=True,
                      help="Path to the predictions JSON file")
    args = parser.parse_args()
    
    print(f"Loading predictions from {args.predictions_file}")
    predictions, sources, references = load_vllm_predictions(args.predictions_file)
    
    print("\nComputing metrics...")
    metrics = compute_metrics(predictions, sources, references)
    
    print("\nResults:")
    print(f"BLEU: {metrics['bleu']:.2f}")
    print(f"chrF: {metrics['chrf']:.2f}")
    print(f"TER: {metrics['ter']:.2f}")
    print(f"COMET: {metrics['comet']:.2f}")

if __name__ == "__main__":
    main()


