import json
import argparse
import os
from tqdm import tqdm

try:
    from google.cloud import translate
except ImportError:
    print("Installing google-cloud-translate...")
    os.system('pip install --quiet google-cloud-translate')
    from google.cloud import translate

try:
    import deepl
except ImportError:
    print("Installing deepl...")
    os.system('pip install --quiet deepl')
    import deepl

def load_test_data(file_path, max_samples=500):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data[:max_samples]  # Only take first 500 examples

def translate_google(text, project_id):
    client = translate.TranslationServiceClient()
    location = "global"
    parent = f"projects/{project_id}/locations/{location}"
    
    response = client.translate_text(
        request={
            "parent": parent,
            "contents": [text],
            "mime_type": "text/plain",
            "source_language_code": "en",
            "target_language_code": "de",
        }
    )
    return response.translations[0].translated_text

def translate_deepl(text, auth_key):
    translator = deepl.Translator(auth_key)
    result = translator.translate_text(text, source_lang="EN", target_lang="DE")
    return result.text

def save_predictions(translations, file_path):
    """Save translations in the correct format with proper character handling"""
    formatted_data = []
    for item in translations:
        # Remove extra quotes and clean special characters
        translation = item["translation"].strip('"')  # Remove surrounding quotes
        
        formatted_item = {
            "translation": {"translation": translation},
            "source": item["source"],
            "reference": item["reference"].replace("<|eot_id|>", "")  # Remove EOT marker
        }
        formatted_data.append(formatted_item)
    
    # Save with proper UTF-8 encoding and without escaping
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(formatted_data, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(formatted_data)} translations to {file_path}")

def translate_batch_deepl(translator, texts, batch_size=50):
    """Translate texts in batches for better efficiency"""
    translations = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        try:
            batch_translations = translator.translate_text(batch, 
                                                        target_lang="DE",
                                                        source_lang="EN")
            translations.extend([t.text for t in batch_translations])
        except Exception as e:
            print(f"Error in batch {i}-{i+batch_size}: {e}")
            # Fallback to single translation for failed batch
            for text in batch:
                try:
                    trans = translator.translate_text(text, 
                                                   target_lang="DE",
                                                   source_lang="EN")
                    translations.append(trans.text)
                except:
                    translations.append("")
    return translations

def process_translations(data, translator):
    texts = [item["prompt"] for item in data]
    total_batches = len(texts) // BATCH_SIZE + (1 if len(texts) % BATCH_SIZE else 0)
    
    with tqdm(total=total_batches, desc="Translating batches") as pbar:
        translations = translate_batch_deepl(translator, texts, BATCH_SIZE)
        pbar.update(1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--google_predictions", type=str, required=True)
    parser.add_argument("--deepl_predictions", type=str, required=True)
    args = parser.parse_args()
    
    # Load first 500 examples from test data
    test_data = load_test_data(args.test_data, max_samples=500)
    print(f"Loaded {len(test_data)} examples from test set")
    
    # Initialize results for both services
    google_translations = []
    deepl_translations = []
    
    # Get API keys
    project_id = os.getenv('GOOGLE_PROJECT_ID')
    auth_key = os.getenv('DEEPL_API_KEY')
    
    # Translate each text with both services
    for item in tqdm(test_data, desc="Translating"):
        source_text = item['input']
        try:
            # Google translation
            google_trans = translate_google(source_text, project_id)
            google_translations.append({
                "translation": google_trans,
                "source": source_text,
                "reference": item['output']
            })
            
            # DeepL translation
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
    save_predictions(google_translations, args.google_predictions)
    save_predictions(deepl_translations, args.deepl_predictions)

if __name__ == "__main__":
    main() 