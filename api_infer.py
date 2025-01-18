import json
import argparse
import os
from tqdm import tqdm
import deepl

def load_test_data(file_path, max_samples=1000):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data[:max_samples]

def translate_deepl(text, auth_key):
    translator = deepl.Translator(auth_key)
    result = translator.translate_text(text, source_lang="EN", target_lang="DE")
    return result.text

def save_predictions(translations, file_path):
    # Write predictions line by line in JSONL format
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in translations:
            prediction = {
                "prompt": f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful AI assistant for translation from English to German. You MUST answer with the following JSON scheme: {{\"translation\":\"Translated text\"}}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{item['source']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
                "predict": json.dumps({"translation": item["translation"]}),
                "label": f"{item['reference']}<|eot_id|>"
            }
            f.write(json.dumps(prediction) + '\n')
    
    print(f"Saved predictions to {file_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--deepl_predictions", type=str, required=True)
    args = parser.parse_args()
    
    # Load first 1000 examples from test data
    test_data = load_test_data(args.test_data, max_samples=1000)
    print(f"Loaded {len(test_data)} examples from test set")
    
    # Initialize results
    deepl_translations = []
    
    # Get API key
    auth_key = os.getenv('DEEPL_API_KEY')
    if not auth_key:
        raise ValueError("DEEPL_API_KEY environment variable not set")
    
    # Translate each text
    for item in tqdm(test_data, desc="Translating"):
        source_text = item['input']
        try:
            deepl_trans = translate_deepl(source_text, auth_key)
            deepl_translations.append({
                "translation": deepl_trans,
                "source": source_text,
                "reference": item['output']
            })
        except Exception as e:
            print(f"Error translating text: {e}")
            continue
    
    # Save results in correct format
    save_predictions(deepl_translations, args.deepl_predictions)

if __name__ == "__main__":
    main() 