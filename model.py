from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import torch
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
import nltk
import re
from sklearn.metrics import accuracy_score

nltk.download('stopwords')

model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
stop_words = set(stopwords.words('english'))

df = pd.read_csv('data/McDonald_s_Reviews.csv', encoding='latin-1')
df['review'] = df['review'].apply(lambda x: ' '.join(term for term in x.split() if term not in stop_words))
df['rating'] = df['rating'].apply(lambda x: int(re.search(r'\d+', x).group()))
df['label'] = (df['rating'] > 3).astype(int)  

train_texts, val_texts, train_labels, val_labels = train_test_split(df['review'], df['label'], test_size=0.2, random_state=42)

train_encodings = tokenizer(train_texts.tolist(), truncation=True, padding=True)
val_encodings = tokenizer(val_texts.tolist(), truncation=True, padding=True)

class McDonaldsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = McDonaldsDataset(train_encodings, train_labels.tolist())
val_dataset = McDonaldsDataset(val_encodings, val_labels.tolist())

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
    }
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,    
    compute_metrics=compute_metrics

)
trainer.train()

eval_result = trainer.evaluate()

print(f"Accuracy on validation set: {eval_result['eval_accuracy']}")
