print("--- Start: v2 Running Sentiment Analysis with 6 Classes")

# Note that this will store a folder "sentiment_model" that when the finished
# model is stored will prevent having to retrain next time the code is run,
# however if the training is not completed then the code will have an error
# next time it is run

import pandas as pd
import os
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from datasets import load_dataset, Dataset
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score
import numpy as np


# Load the dataset
dataset = load_dataset("liar")

# Convert dataset to DataFrame
train_df = pd.DataFrame(dataset['train'])
# train_df.to_csv("AML_Proj_data_liar_huggingface_train.csv")
# pd.DataFrame(dataset['test']).to_csv("AML_Proj_data_liar_huggingface_test.csv")
# pd.DataFrame(dataset['validation']).to_csv("AML_Proj_data_liar_huggingface_validation.csv")

# Creating new column called 'label' with 1 for true and mostly-true values, else 0 i.e. 1=real, 0=fake
train_df['label_binary'] = [1 if x == 3 or x == 2 else 0 for x in train_df['label']]

# Focus on relevant columns and merge them into a single text column
feature_cols = ['subject', 'speaker', 'job_title', 'party_affiliation', 'state_info', 'statement']
train_df['text'] = train_df[feature_cols[:-1]].fillna('').agg(' '.join, axis=1) + " " + train_df['statement']
df = train_df[['text', 'label_binary']]
df.rename(columns = {'label_binary': 'label'}, inplace = True)

# Encode labels
label_dict = {label: i for i, label in enumerate(df['label'].unique())}
df['label'] = df['label'].replace(label_dict)

# Split the data into train and test sets
train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)

# Convert DataFrames to Hugging Face datasets
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

model_path = 'sentiment_model'

# Check if model and tokenizer are already saved
if os.path.exists(model_path):
    tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    model = DistilBertForSequenceClassification.from_pretrained(model_path, num_labels=len(label_dict))
else:
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=len(label_dict))
    tokenizer.save_pretrained(model_path)

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def tokenize(batch):
    return tokenizer(batch['text'], padding="max_length", truncation=True, max_length=512)


# Tokenize the data
train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)


# Define the metric computation
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    precision = precision_score(labels, predictions, average='weighted')
    recall = recall_score(labels, predictions, average='weighted')
    roc_auc = roc_auc_score(labels, predictions)  # Calculate ROC-AUC score
    conf_mat = confusion_matrix(labels, predictions)
    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "roc_auc": roc_auc, # Add ROC-AUC to the metrics
        "confusion_matrix": conf_mat.tolist()  # Convert to list
    }


# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=7,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    do_eval=True,
    evaluation_strategy="epoch"
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

# Train and save the model if not already done
if not os.path.exists(model_path + '/pytorch_model.bin'):
    trainer.train()
    model.save_pretrained(model_path)

# Evaluate the model
results = trainer.evaluate()
print(f"Model evaluation results: {results}")

print("--- End: v2 Sentiment Analysis Completed Successfully")
