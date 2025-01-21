import os
import json
import argparse
from datetime import datetime
import deepl
import sacrebleu
from comet import download_model, load_from_checkpoint

def load_test_data(file_path, max_samples=500):
    """Load up to max_samples from a JSON array with 'source' and 'reference' fields."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data[:max_samples]

def translate_deepl_single(text, auth_key):
    """Translate a single string to German (DE) from English (EN) with DeepL."""
    translator = deepl.Translator(auth_key)
    result = translator.translate_text(text, source_lang="EN", target_lang="DE")
    return result.text

def translate_deepl_batch(texts, auth_key, batch_size=50):
    """Translate a list of strings in batches using DeepL."""
    translator = deepl.Translator(auth_key)
    all_translations = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        try:
            batch_res = translator.translate_text(batch, source_lang="EN", target_lang="DE")
            all_translations.extend([r.text for r in batch_res])
        except Exception as e:
            print(f"Error translating batch {i} - {i + len(batch)}: {str(e)}")
            raise
    
    return all_translations

def compute_metrics(predictions, references):
    """Compute BLEU, chrF, TER, and COMET scores given parallel lists of predictions and references."""
    # sacreBLEU metrics
    bleu = sacrebleu.metrics.BLEU()
    chrf = sacrebleu.metrics.CHRF()
    ter = sacrebleu.metrics.TER()

    bleu_score = bleu.corpus_score(predictions, [references]).score
    chrf_score = chrf.corpus_score(predictions, [references]).score
    ter_score = ter.corpus_score(predictions, [references]).score
    
    # COMET
    print("Downloading COMET model for evaluation...")
    model_path = download_model("Unbabel/wmt22-comet-da")
    comet_model = load_from_checkpoint(model_path)

    comet_data = [
        {"src": "", "mt": pred, "ref": ref}
        for pred, ref in zip(predictions, references)
    ]
    comet_outputs = comet_model.predict(comet_data, batch_size=32, gpus=1)
    comet_score = float(comet_outputs.system_score)

    return {
        "BLEU": bleu_score,
        "chrF": chrf_score,
        "TER": ter_score,
        "COMET": comet_score
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True,
                        help="Path to JSON file with data: each item must have 'input' and 'output'.")
    parser.add_argument("--output_dir", type=str, default='.',
                        help="Directory to save translations and metrics.")
    args = parser.parse_args()
    
    # Read environment variables
    auth_key = os.getenv('DEEPL_API_KEY', '')
    if not auth_key:
        raise ValueError("DEEPL_API_KEY environment variable not set!")
    
    client = os.getenv('CLIENT', 'unknown_client')
    target_lang = os.getenv('TARGET_LANG', 'DE')
    timestamp = os.getenv('TIMESTAMP', datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))

    # Batch size from environment
    batch_size = int(os.getenv('TRANSLATION_BATCH_SIZE', '50'))

    # Load data
    print(f"Loading test data from {args.input_file}...")
    data = load_test_data(args.input_file)
    print(f"Loaded {len(data)} examples.")

    # Extract sources and references using the correct keys from your JSON
    sources = [d["input"] for d in data]
    references = [d["output"] for d in data]

    # Translate with DeepL in batches
    print(f"Translating {len(sources)} texts with batch size = {batch_size}...")
    predictions = translate_deepl_batch(sources, auth_key, batch_size=batch_size)
    print("Translation completed.")

    # Save predictions
    os.makedirs(args.output_dir, exist_ok=True)
    predictions_file = os.path.join(args.output_dir, f"predictions_deepl_{timestamp}.json")

    output_data = []
    for src, ref, pred in zip(sources, references, predictions):
        output_data.append({
            "source": src,
            "reference": ref,
            "prediction": pred
        })

    with open(predictions_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(output_data)} translations to {predictions_file}")

    # Compute metrics (no W&B logging)
    print("Computing evaluation metrics (BLEU, chrF, TER, COMET)...")
    metrics = compute_metrics(predictions, references)
    print("Metrics computed:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.2f}")

    # Save metrics to JSON
    metrics_file = os.path.join(args.output_dir, "evaluation_metrics.json")
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {metrics_file}")

if __name__ == "__main__":
    main()
