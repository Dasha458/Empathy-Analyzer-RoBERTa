import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.nn import CrossEntropyLoss
import os

DATASET_PATH = "empathy.csv"
MODEL_SAVE_PATH = 'model_files/'
MODEL_NAME = 'roberta-base'
MAX_SEQ_LENGTH = 64
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Використовується пристрій: {DEVICE}")

df = pd.read_csv(DATASET_PATH)
df = df.rename(columns={'empathetic_segments': 'text', 'empathy_class ': 'label'})
df = df[['text', 'label']]
df.dropna(subset=['text', 'label'], inplace=True)

if df['label'].dtype == 'object':
    df['label'] = df['label'].astype('category').cat.codes
if df['label'].min() > 0:
    df['label'] = df['label'] - df['label'].min()

num_labels = df['label'].nunique()
print(f"\nКількість унікальних класів (num_labels): {num_labels}")

df_class4 = df[df['label']==4]
df_augmented = pd.concat([df, df_class4]*3, ignore_index=True)
print(f"Розмір після аугментації класу 4: {len(df_augmented)}")

stratify_data = df_augmented['label'] if len(df_augmented) >= num_labels else None
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df_augmented['text'].tolist(),
    df_augmented['label'].tolist(),
    test_size=0.2,
    random_state=42,
    stratify=stratify_data
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=MAX_SEQ_LENGTH)
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=MAX_SEQ_LENGTH)

class EmpathyDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item
    def __len__(self):
        return len(self.labels)

train_dataset = EmpathyDataset(train_encodings, train_labels)
test_dataset = EmpathyDataset(test_encodings, test_labels)

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_labels),
    y=train_labels
)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)
print(f"Class weights: {class_weights}")

from transformers import Trainer

class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, *args, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = CrossEntropyLoss(weight=class_weights)
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)
model.to(DEVICE)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=10,           
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=1e-5,
    logging_dir='./logs',
    logging_steps=50,
    save_strategy='no',
    fp16=(DEVICE.type == 'cuda'),
    dataloader_pin_memory=False,
    report_to="none"
)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted', zero_division=0)
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'precision': precision, 'recall': recall, 'f1': f1}

trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

print(f"\nПОЧАТОК ТРЕНУВАННЯ НА МОДЕЛІ {MODEL_NAME}")
trainer.train()

if not os.path.exists(MODEL_SAVE_PATH):
    os.makedirs(MODEL_SAVE_PATH)

model.save_pretrained(MODEL_SAVE_PATH)
tokenizer.save_pretrained(MODEL_SAVE_PATH)
print("Модель та токенізатор УСПІШНО збережено.")

results = trainer.evaluate()
print("\nМетрики оцінки після тренування:")
for k, v in results.items():
    print(f"{k}: {v:.4f}")

preds = trainer.predict(test_dataset).predictions.argmax(-1)
print("\nClassification Report:")
print(classification_report(test_labels, preds, digits=4))
