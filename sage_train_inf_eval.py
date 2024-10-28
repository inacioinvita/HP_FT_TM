import os
import random
import numpy as np
import torch
import subprocess
import sacrebleu
import pandas as pd
from comet import download_model, load_from_checkpoint
import transformers
import ctranslate2
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
    TrainingArguments, TrainerCallback, TrainerState, TrainerControl, AutoConfig
)
from datasets import Dataset, DatasetDict
from peft import PeftModel, LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
import mlflow
import mlflow.pytorch
import json
import argparse
import datetime
import uuid
from accelerate import Accelerator, DataLoaderConfiguration

# Define the base directory
BASE_DIR = '/home/ivieira/chicago2'

# MLFlow Callback
class MLflowCallback(TrainerCallback):
    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if logs is not None:
            for key, value in logs.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(key, value, step=state.global_step)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_data(train_source, train_target, eval_source, eval_target, test_source, test_target):
    def read_file(file_path):
        with open(file_path, encoding="utf-8") as f:
            return [line.strip() for line in f.readlines()]

    train_source_sentences = read_file(train_source) if train_source else None
    train_target_sentences = read_file(train_target) if train_target else None
    eval_source_sentences = read_file(eval_source) if eval_source else None
    eval_target_sentences = read_file(eval_target) if eval_target else None
    test_source_sentences = read_file(test_source) if test_source else None
    test_target_sentences = read_file(test_target) if test_target else None
    return train_source_sentences, train_target_sentences, eval_source_sentences, eval_target_sentences, test_source_sentences, test_target_sentences

def create_prompt(source_lang, target_lang, sources, targets=None):
    prompts = []
    if targets:
        # Training or Evaluation with target sentences
        template = '''<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful AI assistant for translation from {src} to {tgt}. You MUST answer with the following JSON scheme: {{"translation": "string"}}<|eot_id|><|start_header_id|>user<|end_header_id|>
{source}<|eot_id|><|start_header_id|>assistant<|end_header_id|>{target}<|end_of_text|>'''
        for src, tgt in zip(sources, targets):
            prompt = template.format(src=source_lang, tgt=target_lang, source=src, target=f'{{"translation": "{tgt}"}}')
            prompts.append(prompt)
    else:
        # Inference without target sentences
        template = '''<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful AI assistant for translation from {src} to {tgt}. You MUST answer with the following JSON scheme: {{"translation": "string"}}<|eot_id|><|start_header_id|>user<|end_header_id|>
{source}<|eot_id|><|start_header_id|>assistant<|end_header_id|>'''
        for src in sources:
            prompt = template.format(src=source_lang, tgt=target_lang, source=src)
            prompts.append(prompt)
    return prompts

def prepare_dataset(prompts, num_train_records=None):
    dataset_dict = {}
    if num_train_records:
        dataset_dict["train"] = Dataset.from_dict({"text": prompts[:num_train_records]})
    else:
        dataset_dict["train"] = Dataset.from_dict({"text": prompts})
    return DatasetDict(dataset_dict)

def load_model_and_tokenizer(model_path, quant_config=None):
    # Check if there's a merged_model subdirectory
    merged_model_path = os.path.join(model_path, 'merged_model')
    if os.path.exists(merged_model_path):
        model_path = merged_model_path
        print(f"Found merged_model subdirectory. Using path: {model_path}")
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map='auto',
            quantization_config=quant_config,
            use_cache=False
        )
    except OSError as e:
        print(f"Error loading model: {e}")
        print("Available files in the model directory:")
        print(os.listdir(model_path))
        raise

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        add_bos_token=True,
        add_eos_token=False
    )
    # Add special tokens if necessary
    special_tokens_dict = {}
    if tokenizer.bos_token is None:
        special_tokens_dict['bos_token'] = '<s>'
    if special_tokens_dict:
        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
        model.resize_token_embeddings(len(tokenizer))
        print(f"Added {num_added_toks} special tokens to the tokenizer and resized model embeddings.")
    
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return model, tokenizer

def translate_batch(prompts, tokenizer, generator, max_length, end_token, topk=1):
    print("Starting batch translation...")
    tokenized_inputs = tokenizer(prompts, return_tensors='pt', padding=True)
    input_ids = tokenized_inputs['input_ids'].tolist()

    results = generator.generate_batch(
        input_ids,
        sampling_topk=topk,
        max_length=max_length,
        min_length=1,
        include_prompt_in_result=False,
        end_token=end_token,
        batch_type="tokens",
        max_batch_size=8096
    )

    translations = [tokenizer.decode(ids, skip_special_tokens=True) for ids in results.sequences_ids]      
    print("Batch translation completed.")
    return translations

def calculate_comet(data_path, source_sentences, translations, references):
    df = pd.DataFrame({"src": source_sentences, "mt": translations, "ref": references})
    data = df.to_dict('records')

    model_dir = os.path.join(data_path, "models/wmt22-comet-da/checkpoints")
    model_path = os.path.join(model_dir, "model.ckpt")

    if not os.path.exists(model_path):
        model_path = download_model("Unbabel/wmt22-comet-da", saving_directory=model_dir)

    model = load_from_checkpoint(model_path)

    seg_scores, sys_score = model.predict(data, batch_size=128, gpus=1).values()
    comet = round(sys_score * 100, 2)
    print("COMET:", comet)
    return comet

def calculate_metrics(data_path, source_sentences, translations, references):
    bleu = sacrebleu.corpus_bleu(translations, [references]).score
    chrf = sacrebleu.corpus_chrf(translations, [references], word_order=2).score
    ter = sacrebleu.metrics.TER().corpus_score(translations, [references]).score
    comet = calculate_comet(data_path, source_sentences, translations, references)
    return round(bleu, 2), round(chrf, 2), round(ter, 2), comet

def perform_evaluation(data_dir, test_source, test_target, translations_file, target_lang, args):
    with open(test_source, encoding="utf-8") as source, open(test_target, encoding="utf-8") as target:     
        source_sentences = [line.strip() for line in source.readlines()]
        target_sentences = [line.strip() for line in target.readlines()]

    with open(translations_file, encoding="utf-8") as translated:
        translations = [line.strip() for line in translated.readlines()]

    bleu, chrf, ter, comet = calculate_metrics(data_dir, source_sentences, translations, target_sentences) 

    with mlflow.start_run(run_name="Evaluation"):
        mlflow.log_param("test_source_file", test_source)
        mlflow.log_param("test_target_file", test_target)
        mlflow.log_param("translations_file", translations_file)
        mlflow.log_param("dataset_size", len(source_sentences))
        mlflow.log_param("target_lang", target_lang)
        mlflow.log_param("model_path", args.model_path)
        mlflow.log_param("tokenizer_path", args.model_path)  # Assuming tokenizer path is same as model path
        mlflow.log_param("ct2_quantization", args.ct2_quantization)
        mlflow.log_param("ct2_compute_type", args.ct2_compute_type)

        mlflow.log_metric("BLEU", bleu)
        mlflow.log_metric("chrF++", chrf)
        mlflow.log_metric("TER", ter)
        mlflow.log_metric("COMET", comet)

        evaluation_results = {
            "BLEU": bleu,
            "chrF++": chrf,
            "TER": ter,
            "COMET": comet
        }
        df = pd.DataFrame([evaluation_results])
        evaluation_results_file = os.path.join(data_dir, "evaluation_results.csv")
        df.to_csv(evaluation_results_file, index=False)
        mlflow.log_artifact(evaluation_results_file)
        mlflow.log_artifact(translations_file)

        print("Evaluation results:")
        print(df)

    return bleu, chrf, ter, comet

def train_model(model, tokenizer, dataset, output_directory, target_lang, num_train_records, args):        
    peft_config = LoraConfig(
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        r=args.lora_r,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)

    training_args = TrainingArguments(
        output_dir=output_directory,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        evaluation_strategy=args.evaluation_strategy,
        eval_steps=args.eval_steps,
        save_total_limit=args.save_total_limit,
        learning_rate=args.learning_rate,
        bf16=args.bf16,
        lr_scheduler_type=args.lr_scheduler_type,
        report_to="mlflow",
    )

    with mlflow.start_run(run_name="Training"):
        mlflow.log_param("num_train_epochs", training_args.num_train_epochs)
        mlflow.log_param("train_batch_size", training_args.per_device_train_batch_size)
        mlflow.log_param("eval_batch_size", training_args.per_device_eval_batch_size)
        mlflow.log_param("learning_rate", training_args.learning_rate)
        mlflow.log_param("lora_alpha", peft_config.lora_alpha)
        mlflow.log_param("lora_dropout", peft_config.lora_dropout)
        mlflow.log_param("lora_r", peft_config.r)
        mlflow.log_param("target_language", target_lang)
        mlflow.log_param("num_train_records", num_train_records)
        # mlflow.log_param("seed", args.seed)
        mlflow.log_param("output_directory", output_directory)
        mlflow.log_param("train_source", args.train_source)
        mlflow.log_param("train_target", args.train_target)
        mlflow.log_param("eval_source", args.eval_source)
        mlflow.log_param("eval_target", args.eval_target)
        mlflow.log_param("model_path", args.model_path)
        mlflow.log_param("checkpoint_dir", args.checkpoint_dir)
        mlflow.log_param("load_in_4bit", args.load_in_4bit)
        mlflow.log_param("bnb_4bit_quant_type", args.bnb_4bit_quant_type)
        mlflow.log_param("bnb_4bit_use_double_quant", args.bnb_4bit_use_double_quant)
        mlflow.log_param("ct2_quantization", args.ct2_quantization)
        mlflow.log_param("ct2_compute_type", args.ct2_compute_type)
        mlflow.log_param("eval_steps", args.eval_steps)
        mlflow.log_param("full", args.full)
        mlflow.log_param("logging_steps", args.logging_steps)
        mlflow.log_param("save_steps", args.save_steps)
        mlflow.log_param("evaluation_strategy", args.evaluation_strategy)
        mlflow.log_param("save_total_limit", args.save_total_limit)

        # Update model configuration for checkpointing
        model.config.use_cache = False
        model.config.gradient_checkpointing = True
        model.config.use_reentrant = False

        # Update Accelerator initialization
        accelerator = Accelerator(
            dataloader_config=DataLoaderConfiguration(
                dispatch_batches=None,
                split_batches=False
            )
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
            callbacks=[MLflowCallback()],
            max_seq_length=512,  # Set max_seq_length to 512
        )

        trainer.train()

        # Merge and save the model
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(os.path.join(output_directory, "merged_model"))
        tokenizer.save_pretrained(os.path.join(output_directory, "merged_model"))

        logs = trainer.state.log_history
        detailed_logs_path = os.path.join(output_directory, "detailed_logs.json")
        with open(detailed_logs_path, "w") as log_file:
            json.dump(logs, log_file, indent=2)
        mlflow.log_artifact(detailed_logs_path)

        # Add error handling and default values
        try:
            final_train_loss = next(log['loss'] for log in reversed(logs) if 'loss' in log)
        except StopIteration:
            final_train_loss = None
            print("Warning: No training loss found in logs.")

        try:
            final_eval_loss = next(log['eval_loss'] for log in reversed(logs) if 'eval_loss' in log)
        except StopIteration:
            final_eval_loss = None
            print("Warning: No evaluation loss found in logs.")

        report = {
            "output_directory": output_directory,
            "total_steps": trainer.state.global_step,
            "final_train_loss": final_train_loss,
            "final_eval_loss": final_eval_loss,
            "epochs_completed": trainer.state.epoch,
        }
        report_path = os.path.join(output_directory, "experiment_report.json")
        with open(report_path, "w") as report_file:
            json.dump(report, report_file, indent=2)
        mlflow.log_artifact(report_path)

        print("Experiment Report:")
        print(json.dumps(report, indent=2))

        print(f"Number of training samples: {len(dataset['train'])}")
        print(f"Number of evaluation samples: {len(dataset['validation'])}")
        print(f"Training batch size: {trainer.args.per_device_train_batch_size * trainer.args.n_gpu}")
        print(f"Gradient accumulation steps: {trainer.args.gradient_accumulation_steps}")
        print(f"Effective batch size: {trainer.args.per_device_train_batch_size * trainer.args.n_gpu * trainer.args.gradient_accumulation_steps}")

def evaluate_model(data_dir, model_path, tokenizer_path, test_source, test_target, translations_file, target_lang, args):
    quant_config = BitsAndBytesConfig(
        load_in_4bit=args.load_in_4bit.lower() == 'true',
        bnb_4bit_quant_type=args.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant.lower() == 'true',
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model, tokenizer = load_model_and_tokenizer(model_path, quant_config)

    # Check if there's a merged_model subdirectory
    merged_model_path = os.path.join(model_path, 'merged_model')
    if os.path.exists(merged_model_path):
        model_path = merged_model_path
        print(f"Using merged_model path for conversion: {model_path}")

    ct2_save_directory = os.path.join(data_dir, "ct2_model")
    os.makedirs(ct2_save_directory, exist_ok=True)
    print(f"Converting model to CTranslate2 format and saving to {ct2_save_directory}...")
    subprocess.run([
        "ct2-transformers-converter",
        "--model", model_path,
        "--quantization", args.ct2_quantization,
        "--output_dir", ct2_save_directory,
        "--force"
    ], check=True, text=True)
    print("Conversion to CTranslate2 format completed.")

    generator = ctranslate2.Generator(ct2_save_directory, device="cuda", compute_type=args.ct2_compute_type)
    print("CT2 Model loaded.")

    with open(test_source, encoding="utf-8") as source, open(test_target, encoding="utf-8") as target:     
        source_sentences = [line.strip() for line in source.readlines()]
        target_sentences = [line.strip() for line in target.readlines()]

    prompts = create_prompt("English", "Portuguese", source_sentences)
    translations = translate_batch(prompts, tokenizer, generator, max_length=args.max_length, end_token='}', topk=1)

    translations_output_dir = os.path.join(data_dir, "results", "inference")
    os.makedirs(translations_output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    unique_id = uuid.uuid4().hex[:8]
    translations_file_name_raw = os.path.join(translations_output_dir, f"final_{target_lang}_test_RAW_translation_output_{timestamp}_{unique_id}.txt")
    with open(translations_file_name_raw, "w+", encoding="utf-8") as f:
        for translation in translations:
            f.write(translation + "\n")
    print(f"Raw translation output saved to {translations_file_name_raw}")

    translations_file_name = os.path.join(translations_output_dir, f"final_{target_lang}_test_translations_{timestamp}_{unique_id}.txt")
    with open(translations_file_name, "w+", encoding="utf-8") as f:
        for translation in translations:
            cleaned_translation = translation.replace('{"translation": "', '').rstrip('"}').replace('\n', ' ')
            f.write(cleaned_translation + "\n")
    print(f"Translations saved to {translations_file_name}")

    bleu, chrf, ter, comet = calculate_metrics(data_dir, source_sentences, translations, target_sentences) 

    with mlflow.start_run(run_name="Inference_Evaluation"):
        mlflow.log_param("test_source_file", test_source)
        mlflow.log_param("test_target_file", test_target)
        mlflow.log_param("translations_file", translations_file_name)
        mlflow.log_param("dataset_size", len(source_sentences))
        mlflow.log_param("target_lang", target_lang)
        mlflow.log_param("model_path", model_path)
        mlflow.log_param("tokenizer_path", tokenizer_path)
        mlflow.log_param("ct2_quantization", args.ct2_quantization)
        mlflow.log_param("ct2_compute_type", args.ct2_compute_type)

        mlflow.log_metric("BLEU", bleu)
        mlflow.log_metric("chrF++", chrf)
        mlflow.log_metric("TER", ter)
        mlflow.log_metric("COMET", comet)

        evaluation_results = {
            "BLEU": bleu,
            "chrF++": chrf,
            "TER": ter,
            "COMET": comet
        }
        df = pd.DataFrame([evaluation_results])
        evaluation_results_file = os.path.join(data_dir, "evaluation_results.csv")
        df.to_csv(evaluation_results_file, index=False)
        mlflow.log_artifact(evaluation_results_file)
        mlflow.log_artifact(translations_file_name)

        print("Evaluation results:")
        print(df)

    return bleu, chrf, ter, comet

def convert_to_ct2(merged_model_path, ct2_output_path, quantization):
    subprocess.run([
        "ct2-transformers-converter",
        "--model", merged_model_path,
        "--quantization", quantization,
        "--output_dir", ct2_output_path,
        "--force"
    ], check=True, text=True)

def main(args):
    data_dir = args.data_dir or BASE_DIR

    train_source_sentences, train_target_sentences, eval_source_sentences, eval_target_sentences, test_source_sentences, test_target_sentences = load_data(
        train_source=args.train_source,
        train_target=args.train_target,
        eval_source=args.eval_source,
        eval_target=args.eval_target,
        test_source=args.test_source,
        test_target=args.test_target
    )

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:///home/ivieira/chicago2/mlruns"))
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "Default_Experiment"))

    quant_config = BitsAndBytesConfig(
        load_in_4bit=args.load_in_4bit.lower() == 'true',
        bnb_4bit_quant_type=args.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant.lower() == 'true',
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model, tokenizer = load_model_and_tokenizer(args.model_path, quant_config)

    if args.checkpoint_dir:
        model = PeftModel.from_pretrained(
            model,
            args.checkpoint_dir,
            repo_type="adapter"
        )
        model = model.merge_and_unload()
        print(f"Loaded and merged checkpoint from {args.checkpoint_dir}")
    elif args.train:
        print("Training from the base model without checkpoint.")
    else:
        print("No checkpoint provided and training flag not set.")

    if args.train and train_source_sentences and train_target_sentences and eval_source_sentences and eval_target_sentences:
        prompts = create_prompt("English", args.target_lang, train_source_sentences, train_target_sentences)
        eval_prompts = create_prompt("English", args.target_lang, eval_source_sentences, eval_target_sentences)

        dataset = DatasetDict({
            "train": Dataset.from_dict({"text": prompts[:args.num_train_records]}),
            "validation": Dataset.from_dict({"text": eval_prompts})
        })

        fine_tuned_dir = os.path.join(data_dir, "models", "fine_tuned_models")
        os.makedirs(fine_tuned_dir, exist_ok=True)
        model_output_name = f"llama-3-8B-{args.target_lang}-{args.num_train_records}{'_full' if args.full else ''}"
        training_output_dir = os.path.join(fine_tuned_dir, model_output_name)
        os.makedirs(training_output_dir, exist_ok=True)

        train_model(model, tokenizer, dataset, args.output_dir, args.target_lang, args.num_train_records, args)

    if args.infer:
        infer_output_dir = os.path.join(data_dir, "results", "inference")
        os.makedirs(infer_output_dir, exist_ok=True)
        evaluate_model(data_dir, args.model_path, args.model_path, args.test_source, args.test_target, args.translations_file, args.target_lang, args)

    if args.evaluate:
        results = evaluate_model(
            args.data_dir,
            args.model_path,
            args.model_path,  # Using the same path for tokenizer
            args.test_source,
            args.test_target,
            args.translations_file,
            args.target_lang,
            args
        )
        print("Evaluation Results:", results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Integrated Training, Inference, and Evaluation Script.") 

    # Define arguments
    parser.add_argument('--data_dir', type=str, default=BASE_DIR, help='Base directory for data and models.')
    parser.add_argument('--train', action='store_true', help='Flag to perform training.')
    parser.add_argument('--train_source', type=str, help='Path to the source training file.')
    parser.add_argument('--train_target', type=str, help='Path to the target training file.')
    parser.add_argument('--eval_source', type=str, help='Path to the source evaluation file.')
    parser.add_argument('--eval_target', type=str, help='Path to the target evaluation file.')
    parser.add_argument('--num_train_records', type=int, default=-1, help='Number of records in the training dataset.')
    parser.add_argument('--full', type=str, default='', help='Full or not full training.')
    parser.add_argument('--checkpoint_dir', type=str, default=None, help='Path to the checkpoint directory.')
    parser.add_argument('--target_lang', type=str, help='Target language (e.g., "pt-pt").') 
    parser.add_argument('--model_path', type=str, default=None, help='Path to the pre-trained model.')     
    parser.add_argument('--infer', action='store_true', help='Flag to perform inference.')
    parser.add_argument('--test_source', type=str, help='Path to the source test file.')
    parser.add_argument('--test_target', type=str, help='Path to the target test file.')
    parser.add_argument('--output_dir', type=str, help='Directory to save the fine-tuned model.')
    parser.add_argument('--evaluate', action='store_true', help='Flag to perform evaluation.')
    parser.add_argument('--translations_file', type=str, help='Path to the translations file.')
    parser.add_argument('--num_train_epochs', type=int, default=2, help='Number of training epochs.')      
    parser.add_argument('--train_batch_size', type=int, default=32, help='Training batch size per device.')    
    parser.add_argument('--eval_batch_size', type=int, default=32, help='Evaluation batch size per device.')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate.')
    parser.add_argument('--lora_alpha', type=int, default=16, help='LoRA alpha parameter.')
    parser.add_argument('--lora_dropout', type=float, default=0.1, help='LoRA dropout rate.')
    parser.add_argument('--lora_r', type=int, default=64, help='LoRA r parameter.')
    parser.add_argument('--load_in_4bit', type=str, default='true', help='Load model in 4-bit (true/false).')
    parser.add_argument('--bnb_4bit_quant_type', type=str, default='nf4', help='BNB 4-bit quantization type.')
    parser.add_argument('--bnb_4bit_use_double_quant', type=str, default='true', help='Use double quantization (true/false).')
    parser.add_argument('--ct2_quantization', type=str, default='int8', help='CTranslate2 quantization type.')
    parser.add_argument('--ct2_compute_type', type=str, default='int8', help='CTranslate2 compute type.')  
    parser.add_argument('--warmup_steps', type=int, default=0, help='Number of warmup steps.')
    parser.add_argument('--logging_steps', type=int, default=250, help='Number of logging steps.')
    parser.add_argument('--save_steps', type=int, default=1000, help='Number of save steps.')
    parser.add_argument('--evaluation_strategy', type=str, default='steps', help='Evaluation strategy.')   
    parser.add_argument('--bf16', action='store_true', help='Use bfloat16 precision.')
    parser.add_argument('--lr_scheduler_type', type=str, default='constant', help='Learning rate scheduler type.')
    parser.add_argument('--max_length', type=int, default=512, help='Max generation length.')
    parser.add_argument('--eval_steps', type=int, default=100, help='Number of evaluation steps.')
    parser.add_argument('--save_total_limit', type=int, default=3, help='Limit the total amount of checkpoints. Deletes the older checkpoints in output_dir.')

    args = parser.parse_args()
    main(args)
