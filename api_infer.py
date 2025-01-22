import os
import json
import argparse
from datetime import datetime
import deepl
import sacrebleu
from comet import download_model, load_from_checkpoint

max_samples = 20  # Set to None to load all samples

# Read environment variables
auth_key = os.getenv('DEEPL_API_KEY', '')
if not auth_key:
    raise ValueError("DEEPL_API_KEY environment variable not set!")

client = os.getenv('CLIENT', 'unknown_client')
target_lang = os.getenv('TARGET_LANG', 'DE')
timestamp = os.getenv('TIMESTAMP', datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))


def load_test_data(file_path, max_samples=max_samples):
    """Load up to max_samples from a JSON array with 'input' field"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data[:max_samples]

def translate_deepl_single(text, auth_key):
    f"""Translate a single string to {target_lang} from English (EN) with DeepL."""
    translator = deepl.Translator(auth_key)
    result = translator.translate_text(text, source_lang="EN", target_lang=target_lang)
    return result.text

def translate_deepl_batch(texts, auth_key, batch_size=50):
    """Translate a list of strings in batches using DeepL."""
    translator = deepl.Translator(auth_key)
    all_translations = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        try:
            batch_res = translator.translate_text(batch, source_lang="EN", target_lang=target_lang)
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

def clean_prediction(text):
    """Clean up extra quotes from predictions"""
    # Remove double quotes at start and end
    text = text.strip('"')
    # Remove Japanese/Chinese quotes at start
    text = text.lstrip('「')
    # Remove Japanese/Chinese quotes at end
    text = text.rstrip('」')
    # Remove any remaining double quotes at start/end
    text = text.strip('"')
    return text

def prepare_text_for_translation(input_data):
    """Extract clean text from input data"""
    # Get the input text and remove outer quotes manually
    text = input_data['input'].strip('"')
    # Unescape forward slashes if present
    text = text.replace('\/', '/')
    return text

def extract_reference(output_data):
    """Extract reference translation from output JSON with fallback"""
    # Remove outer quotes
    output_text = output_data['output'].strip('"')
    
    try:
        # Try to parse as JSON first
        output_json = json.loads(output_text)
        reference = output_json.get('translation', output_text)
    except json.JSONDecodeError:
        # Fallback: use the raw output text
        reference = output_text
    
    # Clean up the reference
    reference = reference.strip('"').replace('\/', '/')
    return reference

def save_translations_with_verify(translations, sources, references, filename):
    data = []
    for idx, (translation, source, reference) in enumerate(zip(translations, sources, references)):
        # Clean quotes from all fields
        clean_source = source.strip('"')
        clean_reference = reference.strip('"')
        data.append({
            "id": idx,
            "prediction": translation,
            "source": clean_source,
            "reference": clean_reference
        })
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        if os.path.exists(filename):
            with open(filename, 'r', encoding='utf-8') as f:
                verify_data = json.load(f)
            print(f"Successfully saved {len(verify_data)} translations to {filename}")
        else:
            print(f"ERROR: File {filename} was not created!")
            
    except Exception as e:
        print(f"Error saving translations: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True,
                       help='Path to input JSON file with test data')
    parser.add_argument('--output_dir', type=str, default='.',
                       help='Directory to save translations')
    args = parser.parse_args()

    print(f"Loading test data from {args.input_file}...")
    test_data = load_test_data(args.input_file, max_samples=max_samples)  # Enforce limit
    print(f"Loaded {len(test_data)} test examples")
    texts = [prepare_text_for_translation(item) for item in test_data]
    print(f"First text to translate: {texts[0]}")

    # Extract sources and references using the correct keys from your JSON
    sources = [prepare_text_for_translation(item) for item in test_data]
    references = [extract_reference(item) for item in test_data]

    # Batch size from environment
    batch_size = int(os.getenv('TRANSLATION_BATCH_SIZE', '50'))

    # Translate with DeepL in batches
    print(f"Translating {len(sources)} texts with batch size = {batch_size}...")
    predictions = translate_deepl_batch(texts, auth_key, batch_size=batch_size)
    print("Translation completed.")

    # Save predictions
    os.makedirs(args.output_dir, exist_ok=True)
    predictions_file = os.path.join(args.output_dir, f"{client}_{target_lang}_predictions_deepl_{timestamp}.json")

    save_translations_with_verify(predictions, sources, references, predictions_file)

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
