from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os

train_df = pd.read_csv('data/amazon_products_train.csv')
valid_df = pd.read_csv('data/amazon_products_valid.csv')

def format_data(row):
    return f"Product: {row['product']} | Description: {row['description']} | Ad: {row['ad']}"

train_df['formatted_text'] = train_df.apply(format_data, axis=1)
valid_df['formatted_text'] = valid_df.apply(format_data, axis=1)

train_dataset = Dataset.from_pandas(train_df[['formatted_text']])
valid_dataset = Dataset.from_pandas(valid_df[['formatted_text']])

model_name = "distilgpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = GPT2LMHeadModel.from_pretrained(model_name)

def tokenize_function(examples):
    tokenized_inputs = tokenizer(examples['formatted_text'], padding="max_length", truncation=True, max_length=128, return_tensors="pt")
    tokenized_inputs['labels'] = tokenized_inputs['input_ids'].clone()
    return tokenized_inputs

tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_valid_dataset = valid_dataset.map(tokenize_function, batched=True)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    predictions = predictions.flatten()
    labels = labels.flatten()
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average='macro')
    recall = recall_score(labels, predictions, average='macro')
    f1 = f1_score(labels, predictions, average='macro')
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=5,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=500,
    evaluation_strategy="epoch",
    prediction_loss_only=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_valid_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

model.save_pretrained("./gpt2_finetuned_amazon")
tokenizer.save_pretrained("./gpt2_finetuned_amazon")

metrics = trainer.evaluate()

print("Metrics after training:")
for key, value in metrics.items():
    print(f"{key}: {value}")

metrics_file_path = os.path.join(training_args.output_dir, "training_metrics.txt")
with open(metrics_file_path, "w") as metrics_file:
    for key, value in metrics.items():
        metrics_file.write(f"{key}: {value}\n")

print("Model został wytrenowany i zapisany w folderze './gpt2_finetuned_amazon'")
print(f"Metryki zostały zapisane w pliku {metrics_file_path}")