from pathlib import Path
from datasets import Dataset
import argparse
from sklearn.model_selection import train_test_split
from transformers import (AutoTokenizer,
                          AutoModelForMaskedLM,
                          DataCollatorForLanguageModeling,
                          Trainer,
                          TrainingArguments,
                          EarlyStoppingCallback)



def run_dapt(corpus_path: str,
             model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
             output_dir: str = "models/bert/dapt_minilm",
             num_train_epochs: int = 3,
             per_device_train_batch_size: int = 32,
             learning_rate: float = 5e-5,
             warmup_steps: int = 0,
             save_total_limit: int = 2,
             logging_steps: int = 100,
             max_seq_length: int = 256,
             val_split: float = 0.1,
             early_stopping_patience: int = 2,
             early_stopping_threshold: float = 0.01,
             save_best_only: bool = True):
    """
    Runs Domain-Adaptive Pretraining (DAPT) on a given text corpus.

    Args:
        corpus_path (str): Path to the text corpus file.
        model_name (str): Name of the pre-trained BERT model to use. default: "sentence-transformers/all-MiniLM-L6-v2".
        output_dir (str): Directory to save the trained model. default: "models/bert/dapt_minilm".
        num_train_epochs (int): Number of training epochs. default: 3.
        per_device_train_batch_size (int): Batch size for training. default: 32.
        learning_rate (float): Learning rate for training. default: 5e-5.
        warmup_steps (int): Number of warmup steps for training. default: 0.
        save_total_limit (int): Number of checkpoints to save. default: 2.
        logging_steps (int): Number of steps to log. default: 100.
        max_seq_length (int): Maximum sequence length for input. default: 256.
        val_split (float): Fraction of the data to use for validation. default: 0.1.
        early_stopping_patience (int): Number of epochs to wait for improvement before early stopping. default: 2.
        early_stopping_threshold (float): Threshold for early stopping improvement. default: 0.01.
        save_best_only (bool): Whether to save only the best model. default: True.

    Returns:
        output_dir (str): Path to the trained model directory.
    """

    # Load dataset from text file bypassing any future caching errors.
    with open(corpus_path, encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]

    if val_split > 0:
        # Train/validation split
        train_texts, val_texts = train_test_split(lines, test_size=val_split, random_state=42)
        dataset = Dataset.from_dict({"text": train_texts})
        val_dataset = Dataset.from_dict({"text": val_texts})
    else:
        # Use full data for training
        dataset = Dataset.from_dict({"text": lines})
        val_dataset = None

    # Tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)

    # Tokenization function
    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=max_seq_length,
        )

    tokenized_train = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    tokenized_val = val_dataset.map(tokenize_fn, batched=True, remove_columns=["text"]) if val_dataset else None

    # Data collator with dynamic masking
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15,
    )

    # Base training arguments
    training_args = {
        "output_dir": output_dir,
        "overwrite_output_dir": True,
        "num_train_epochs": num_train_epochs,
        "per_device_train_batch_size": per_device_train_batch_size,
        "save_total_limit": save_total_limit,
        "prediction_loss_only": True,
        "logging_steps": logging_steps,
        "learning_rate": learning_rate,
        "warmup_steps": warmup_steps,
        "save_strategy": "epoch",
        "report_to": "none",
    }

    # Add validation-related args only if val_split > 0
    if val_dataset:
        training_args.update({
            "eval_strategy": "epoch",
            "load_best_model_at_end": save_best_only,
            "metric_for_best_model": "eval_loss",
            "greater_is_better": False,
        })


    training_args = TrainingArguments(**training_args)

    # Callbacks
    callbacks = []
    if val_dataset and early_stopping_patience > 0:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=early_stopping_patience,
                early_stopping_threshold=early_stopping_threshold,
            )
        )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val if val_dataset else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks if val_dataset else None,
    )

    # Train
    print("🚀 Starting Domain-Adaptive Pretraining (DAPT)...")
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"✅ DAPT finished! Model saved at: {output_dir}")
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Domain-Adaptive Pretraining (DAPT) for BERT/SBERT")

    parser.add_argument("--model_name", type=str, default="sentence-transformers/all-MiniLM-L6-v2",
                        help="Pretrained model name or path to load")
    parser.add_argument("--corpus_path", type=str, default="data/processed/domain_corpus.txt",
                        help="Path to plain text corpus for DAPT")
    parser.add_argument("--output_dir", type=str, default="models/dapt_bert",
                        help="Directory to save the fine-tuned model")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Training batch size per device")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Learning rate for AdamW optimizer")
    parser.add_argument("--warmup_steps", type=int, default=0,
                        help="Number of warmup steps for LR scheduler")
    parser.add_argument("--max_seq_length", type=int, default=256,
                        help="Maximum sequence length for inputs")
    parser.add_argument("--val_split", type=float, default=0.1,
                        help="Fraction of data to use for validation (set 0 for no validation)")
    parser.add_argument("--early_stopping_patience", type=int, default=2,
                        help="Number of evals with no improvement before stopping (ignored if val_split=0)")
    parser.add_argument("--early_stopping_threshold", type=float, default=0.01,
                        help="Minimum improvement in eval loss to be considered progress (ignored if val_split=0)")
    parser.add_argument("--save_best_only", action="store_true",
                        help="Save only the best checkpoint (ignored if val_split=0)")

    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    run_dapt(
        model_name=args.model_name,
        corpus_path=args.corpus_path,
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        max_seq_length=args.max_seq_length,
        val_split=args.val_split,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_threshold=args.early_stopping_threshold,
        save_best_only=args.save_best_only,
    )


if __name__ == "__main__":
    main()
