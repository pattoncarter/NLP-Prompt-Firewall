import pandas as pd
from datasets import load_dataset, DatasetDict
RANDOM_SEED = 42

# Load the two datasets
df_benign = pd.read_csv("benign_dataset.csv")
df_malicious = pd.read_csv("malicious_dataset(in).csv")

df_full = pd.concat([df_benign, df_malicious])

df_full = df_full.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

df_full.to_csv("full_dataset.csv", index=False)

full_dataset = load_dataset('csv', data_files='full_dataset.csv')

# Split into train and validation sets (80/20 split)
train_test_split = full_dataset['train'].train_test_split(test_size=0.2, seed=RANDOM_SEED)

dataset_dict = DatasetDict({
    'train': train_test_split['train'],
    'validation': train_test_split['test']
})

print(dataset_dict)

# tokenization
import importlib.metadata as m; print(m.version("transformers"))
from transformers import AutoTokenizer
# Load tokenizer for DistilBERT
model_checkpoint = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

def tokenize_function(examples):
    return tokenizer(examples["prompt"], padding="max_length", truncation=True)

tokenized_datasets = dataset_dict.map(tokenize_function, batched=True)

tokenized_datasets = tokenized_datasets.remove_columns(["prompt", "category"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

# evaluate metrics
import numpy as np
import evaluate

metric_precision = evaluate.load("precision")
metric_recall = evaluate.load("recall")
metric_f1 = evaluate.load("f1")
metric_accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    precision = metric_precision.compute(predictions=predictions, references=labels, pos_label=1)["precision"]
    recall = metric_recall.compute(predictions=predictions, references=labels, pos_label=1)["recall"]
    f1 = metric_f1.compute(predictions=predictions, references=labels, pos_label=1)["f1"]
    accuracy = metric_accuracy.compute(predictions=predictions, references=labels)["accuracy"]

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

# fine-tune the model
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint, 
    num_labels=2
)

# set hyperparams
training_args = TrainingArguments(
    output_dir="my_llm_firewall_model",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False,
    report_to='tensorboard'
)

# create trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)

# run training
print("--- Starting Model Training ---")
trainer.train()
print("--- Training Complete ---")

print("--- Final Evaluation on Validation Set ---")
final_metrics = trainer.evaluate()
print(final_metrics)

# save final model
trainer.save_model("./my_final_firewall_model")