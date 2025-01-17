import json
import sacrebleu
from comet import download_model, load_from_checkpoint
import torch
import os
import re

def clean_json_output(text):
    """Clean and normalize JSON string before parsing"""
    # Skip if already valid JSON
    try:
        json.loads(text)
        return text
    except:
        pass
    
    # Fix German quotes and nested quotes
    text = text.replace('„', '"').replace('"', '"').replace('"', '"')
    
    # Handle quotes within translation text by escaping them
    if '"translation"' in text:
        start = text.find('"translation"') + len('"translation"')
        colon_pos = text.find(':', start)
        if colon_pos != -1:
            before = text[:colon_pos+1]
            content = text[colon_pos+1:]
            # Clean up the content
            content = content.strip()
            if content.startswith('"'):
                content = content[1:]
            if content.endswith('"'):
                content = content[:-1]
            # Escape any remaining quotes in content
            content = content.replace('"', '\\"')
            # Rebuild with proper JSON structure
            text = f'{before} "{content}"'
    
    # Ensure proper JSON structure
    if not text.startswith('{'):
        text = '{' + text
    if not text.endswith('}'):
        text = text + '}'
        
    # Remove any trailing characters after JSON
    try:
        end = text.rindex('}') + 1
        text = text[:end]
    except ValueError:
        pass
    
    return text

def load_vllm_predictions(predictions_file):
    predictions = []
    sources = []
    references = []
    
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
                
                # Parse prediction with cleaning
                try:
                    pred_str = data['predict']
                    cleaned_pred = clean_json_output(pred_str)
                    pred_json = json.loads(cleaned_pred)
                except Exception as e:
                    print(f"\nError parsing prediction at line {i}:")
                    print(f"Raw prediction: {data['predict']}")
                    raise e
                
                # Parse reference
                ref_json = json.loads(data['label'].replace('<|eot_id|>', ''))
                
                predictions.append(pred_json['translation'])
                sources.append(source)
                references.append(ref_json['translation'])
                
            except Exception as e:
                print(f"Warning: Error processing line {i}: {e}")
                continue
    
    if not predictions:
        raise ValueError("No valid predictions found in file")
    
    print(f"\nSuccessfully loaded {len(predictions)} prediction triplets")
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

def extract_translation(output_text):
    """Extract the *last* valid JSON block containing 'translation' from the model output."""
    print("Raw output:", output_text)  # keep for debugging

    # Find all JSON-like patterns
    pattern = r"\{[^{}]*\}"
    candidates = re.findall(pattern, output_text)
    
    # Check candidates from last to first (since example appears first in prompt)
    for candidate in reversed(candidates):
        try:
            parsed = json.loads(candidate)
            if "translation" in parsed:
                translation = parsed["translation"].strip()
                if translation and translation != "Translated text":  # Skip example
                    print("Found valid JSON:", candidate)
                    return translation
        except json.JSONDecodeError:
            continue

    print("No valid translation JSON found")
    return ""

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


