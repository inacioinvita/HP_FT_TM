import os
import json
import torch
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from transformers import BitsAndBytesConfig


# Define base directory
BASE_DIR = '/home/ivieira/chicago2'


def load_data(train_file, eval_file):
    """Loads training and evaluation data from files."""
    train_file = os.path.join(BASE_DIR, train_file) if not os.path.isabs(train_file) else train_file
    eval_file = os.path.join(BASE_DIR, eval_file) if not os.path.isabs(eval_file) else eval_file
    print(f"Loading training data from: {train_file}")
    print(f"Loading evaluation data from: {eval_file}")
    with open(train_file, encoding="utf-8") as train, open(eval_file, encoding="utf-8") as evaluation:
        train_sentences = [sent.strip() for sent in train.readlines()]
        eval_sentences = [sent.strip() for sent in evaluation.readlines()]
    return train_sentences, eval_sentences


def create_prompt(source_lang, target_lang, sources, targets, llama_format=True):
    """Generates prompts for training based on source and target sentences."""
    prompts = []
    llama_prompt_format = '''<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful AI assistant for translation from {} to {}. You MUST answer with the following JSON scheme: {{"translation": "string"}}<|eot_id|><|start_header_id|>user<|end_header_id|>

{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>{}<|end_of_text|>'''
    for src, trg in zip(sources, targets):
        source_segment = source_lang + ": " + src
        target_segment = f'{{"translation": "{trg}"}}'
        segment = llama_prompt_format.format(source_lang, target_lang, source_segment, target_segment)
        prompts.append(segment)
    return prompts


def prepare_dataset(prompts, eval_prompts, num_train_records):
    """Prepares dataset for training."""
    return DatasetDict({
        "train": Dataset.from_dict({"text": prompts[:num_train_records]}),
        "validation": Dataset.from_dict({"text": eval_prompts})
    })


def load_model_and_tokenizer(model_path):
    """Loads the model and tokenizer with 4-bit quantization."""
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map='auto',
        quantization_config=nf4_config,
        use_cache=False
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        add_bos_token=True,
        add_eos_token=False
    )

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer


def train_model(model, tokenizer, dataset, output_directory):
    """Configures and trains the model."""
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)

    training_args = TrainingArguments(
        output_dir=output_directory,
        num_train_epochs=1,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=0.03,
        logging_steps=50,
        save_steps=50,
        evaluation_strategy="steps",
        eval_steps=50,
        learning_rate=1e-3,
        bf16=True,
        lr_scheduler_type='constant',
    )

    trainer = SFTTrainer(
        model=model,
        peft_config=peft_config,
        tokenizer=tokenizer,
        dataset_text_field="text",
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
    )

    trainer.train()

    # Save logs
    logs = trainer.state.log_history
    detailed_logs_path = os.path.join(output_directory, "detailed_logs.json")
    with open(detailed_logs_path, "w") as log_file:
        json.dump(logs, log_file, indent=2)

    print(f"Model saved to: {output_directory}")



def main(train_file, eval_file, target_lang, num_train_records):
    """Main function to execute the training pipeline."""
    source_train_file = train_file.replace('.de', '.en')
    source_eval_file = eval_file.replace('.de', '.en')

    source_lang = "English"

    target_train_sentences, target_eval_sentences = load_data(train_file, eval_file)
    source_train_sentences, source_eval_sentences = load_data(source_train_file, source_eval_file)

    prompts = create_prompt(source_lang, target_lang, source_train_sentences, target_train_sentences)
    eval_prompts = create_prompt(source_lang, target_lang, source_eval_sentences, target_eval_sentences)

    dataset = prepare_dataset(prompts, eval_prompts, num_train_records)

    model_path = os.path.join('~/spinning-storage/ivieira/chicago2/models/llama318b')
    model, tokenizer = load_model_and_tokenizer(model_path)

    output_directory = os.path.join(BASE_DIR, 'models', 'fine_tuned_models')
    os.makedirs(output_directory, exist_ok=True)

    train_model(model, tokenizer, dataset, output_directory)


if __name__ == "__main__":
    train_file = "BALS_train_en-de.de"
    eval_file = "BALS_dev_en-de.de"
    target_lang = "German"
    num_train_records = 1000  # Adjust as needed
    main(train_file, eval_file, target_lang, num_train_records)
