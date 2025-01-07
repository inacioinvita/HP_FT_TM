import os
import pandas as pd
import random
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from transformers import Trainer, DataCollatorWithPadding
import torch
from peft import PeftModel, LoraConfig, get_peft_model, prepare_model_for_kbit_training
import json
import argparse
from trl import SFTTrainer

# Define the base directory
BASE_DIR = '/home/ivieira/chicago2'

def load_data(train_file, eval_file):
    if not os.path.isabs(train_file):
        train_file = os.path.join(BASE_DIR, train_file)
    if not os.path.isabs(eval_file):
        eval_file = os.path.join(BASE_DIR, eval_file)
    print(f"Loading training data from: {train_file}")
    print(f"Loading evaluation data from: {eval_file}")
    with open(train_file, encoding="utf-8") as train, open(eval_file, encoding="utf-8") as evaluation:
        train_sentences = [sent.strip() for sent in train.readlines()]
        eval_sentences = [sent.strip() for sent in evaluation.readlines()]
    return train_sentences, eval_sentences


def create_prompt(source_lang, target_lang, new_sources, new_targets, llama_format=True):
    prompts = []
    llama_prompt_format = '''<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful AI assistant for translation from {} to {}. You MUST answer with the following JSON scheme: {{"translation": "string"}}<|eot_id|><|start_header_id|>user<|end_header_id|>

{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>{}<|end_of_text|>'''
    for new_src, new_trg in zip(new_sources, new_targets):
        source_segment = source_lang + ": " + new_src
        target_segment = f'{{"translation": "{new_trg}"}}'
        if llama_format:
            segment = llama_prompt_format.format(source_lang, target_lang, source_segment, target_segment)
        prompts.append(segment)
    return prompts

def prepare_dataset(prompts, eval_prompts, num_train_records):
    return DatasetDict({
        "train": Dataset.from_dict({"text": prompts[:num_train_records]}),
        "validation": Dataset.from_dict({"text": eval_prompts})
    })

def load_model_and_tokenizer(model_path):
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

    special_tokens_dict = {}
    if tokenizer.bos_token is None:
        special_tokens_dict['bos_token'] = '<s>'
    stopping_criteria = tokenizer.convert_ids_to_tokens(tokenizer.encode("assistant/n"))

    if special_tokens_dict:
        tokenizer.add_special_tokens(special_tokens_dict)
        model.resize_token_embeddings(len(tokenizer))

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer

def train_model(model, tokenizer, dataset, output_directory):
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
        num_train_epochs=2,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        warmup_steps=0,
        logging_steps=1000,
        save_steps=1000,
        evaluation_strategy="steps",
        eval_steps=100,
        learning_rate=1e-3,
        bf16=True,
        lr_scheduler_type='constant',
    )

    trainer = SFTTrainer(
        model=model,
        peft_config=peft_config,
        tokenizer=tokenizer,
        packing=True,
        dataset_text_field="text",
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
    )

    trainer.train()

    # Collect logs from the training process
    logs = trainer.state.log_history

    # Save detailed logs
    detailed_logs_path = os.path.join(output_directory, "detailed_logs.json")
    with open(detailed_logs_path, "w") as log_file:
        json.dump(logs, log_file, indent=2)

    # Print model save path
    print(f"Model saved to: {output_directory}")

    # Generate and print a report
    report = {
        "output_directory": output_directory,
        "total_steps": trainer.state.global_step,
        "final_train_loss": logs[-1].get("loss", "N/A"),
        "final_eval_loss": logs[-1].get("eval_loss", "N/A"),
        "epochs_completed": logs[-1].get("epoch", "N/A"),
    }
    report_path = os.path.join(output_directory, "experiment_report.json")
    with open(report_path, "w") as report_file:
        json.dump(report, report_file, indent=2)

    print("Experiment Report:")
    print(json.dumps(report, indent=2))

def main(train_file, eval_file, target_lang, num_train_records, full, checkpoint_dir=None):
    # Correctly construct source file paths
    source_train_file = train_file.replace(f'.{target_lang}', '.en')
    source_eval_file = eval_file.replace(f'.{target_lang}', '.en')

    print(f"Source training data: {source_train_file}")
    print(f"Source evaluation data: {source_eval_file}")

    target_train_sentences, target_eval_sentences = load_data(train_file, eval_file)
    source_train_sentences, source_eval_sentences = load_data(source_train_file, source_eval_file)

    source_lang = "English"

    prompts = create_prompt(source_lang, target_lang, source_train_sentences, target_train_sentences)
    eval_prompts = create_prompt(source_lang, target_lang, source_eval_sentences, target_eval_sentences)

    print(f"Number of prompts: {len(prompts)}")
    for prompt in prompts[40:45]:
        print(prompt + "\n\n")

    model_path = os.path.join(BASE_DIR, 'models', 'meta-llama', 'Meta-Llama-3-8B-Instruct')

    if checkpoint_dir:
        # Load the base model with configuration
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

        # Load the PEFT model
        model = PeftModel.from_pretrained(model, checkpoint_dir)

        # Merge the PEFT model with the base model
        model = model.merge_and_unload()
        print(f"Loaded and merged checkpoint from {checkpoint_dir}")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            add_bos_token=True,
            add_eos_token=False
        )
    else:
        model, tokenizer = load_model_and_tokenizer(model_path)
        print("Tokenizer and model loaded successfully.")

    dataset = prepare_dataset(prompts, eval_prompts, num_train_records)

    print(dataset)

    # Update output directory
    output_directory = os.path.join(BASE_DIR, 'models', 'fine_tuned_models')
    model_output_name = f"llama-3-8B-{target_lang}-{num_train_records}{full}"
    output_directory = os.path.join(output_directory, model_output_name)
    os.makedirs(output_directory, exist_ok=True)

    train_model(model, tokenizer, dataset, output_directory)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare and train a translation model.")
    parser.add_argument('--full', type=str, required=True, help='full or not full')
    parser.add_argument('--train_file', type=str, required=True, help='Path to the target training file')
    parser.add_argument('--eval_file', type=str, required=True, help='Path to the target evaluation file')
    parser.add_argument('--target_lang', type=str, required=True, help='Target language (e.g., "pt-pt")')
    parser.add_argument('--num_train_records', type=int, help='Number of records in the training dataset')
    parser.add_argument('--checkpoint_dir', type=str, default=None, help='Path to the checkpoint directory')

    args = parser.parse_args()
    main(args.train_file, args.eval_file, args.target_lang, args.num_train_records, args.full, args.checkpoint_dir)

