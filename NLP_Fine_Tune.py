import pandas as pd
from datasets import load_dataset, DatasetDict
RANDOM_SEED = 42
# --------------------------------- SHUFFLE AND SPLIT DATASET ---------------------------------
# Load the two datasets
df_benign = pd.read_csv("benign_dataset.csv")
df_malicious = pd.read_csv("malicious_dataset(in).csv")

# Combine them
df_full = pd.concat([df_benign, df_malicious])

# CRITICAL: Shuffle the dataset
df_full = df_full.sample(frac=1).reset_index(drop=True)

# Save to a new file
df_full.to_csv("full_dataset.csv", index=False)

# Load the new file using the 'datasets' library
full_dataset = load_dataset('csv', data_files='full_dataset.csv')

# Split into train and validation sets (80/20 split)
# This creates your 'validation ("hold out") set' for the paper
train_test_split = full_dataset['train'].train_test_split(test_size=0.2, seed=RANDOM_SEED)

# Rename for clarity
dataset_dict = DatasetDict({
    'train': train_test_split['train'],
    'validation': train_test_split['test']
})

print(dataset_dict)

# --------------------------------- Tokenize Data ---------------------------------
import importlib.metadata as m; print(m.version("transformers"))
from transformers import AutoTokenizer
# Load the tokenizer for DistilBERT
model_checkpoint = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

# Add this print statement to be 100% sure
print(f"Successfully loaded tokenizer: {type(tokenizer)}")

# Create a function to tokenize our text
def tokenize_function(examples):
    # 'padding="max_length"' and 'truncation=True' handle
    # prompts that are shorter or longer than the model's max length.
    return tokenizer(examples["prompt"], padding="max_length", truncation=True)

# Apply this function to all splits in our dataset
tokenized_datasets = dataset_dict.map(tokenize_function, batched=True)

# You can now drop the text columns, as they are no longer needed
tokenized_datasets = tokenized_datasets.remove_columns(["prompt", "category"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

# --------------------------------- Evaluate Metrics ---------------------------------
import numpy as np
import evaluate

# Load the metrics from the 'evaluate' library
metric_precision = evaluate.load("precision")
metric_recall = evaluate.load("recall")
metric_f1 = evaluate.load("f1")
metric_accuracy = evaluate.load("accuracy")

# This function will be called by the Trainer at the end of each epoch
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    # Calculate all metrics. 
    # 'pos_label=1' ensures we are measuring the 'malicious' class
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
    
# --------------------------------- Fine-Tune the Model ---------------------------------
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

# Load the pre-trained model with a classification head on top
model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint, 
    num_labels=2
)

# Set hyperparams
training_args = TrainingArguments(
    output_dir="my_llm_firewall_model",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3, # 3 epochs is a good starting point
    weight_decay=0.01,
    eval_strategy="epoch",  # Run validation at the end of each epoch
    save_strategy="epoch",
    load_best_model_at_end=True,  # This will keep the best model
    push_to_hub=False,
    report_to='tensorboard'  # Enable TensorBoard logging
)

# Create trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    processing_class=tokenizer, # changed from tokenizer to processing_class due to future deprication
    compute_metrics=compute_metrics,
)

# fine tune the model
print("--- Starting Model Training ---")
trainer.train()
print("--- Training Complete ---")

# After training, run a final evaluation and print the results
print("--- Final Evaluation on Validation Set ---")
final_metrics = trainer.evaluate()
print(final_metrics)

# Save final model
trainer.save_model("./my_final_firewall_model")