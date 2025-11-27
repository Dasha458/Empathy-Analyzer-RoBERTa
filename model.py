import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import os

DATASET_PATH = "empathy.csv"
MODEL_SAVE_PATH = 'model_files/'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Використовується пристрій: {DEVICE}")

try:
    df = pd.read_csv(DATASET_PATH)
except FileNotFoundError:
    print(f"Помилка: Файл '{DATASET_PATH}' не знайдено.")
    exit()

df = df.rename(columns={'empathetic_segments': 'text', 'empathy_class ': 'label'})
df = df[['text', 'label']]
df.dropna(subset=['text', 'label'], inplace=True)

if df['label'].dtype == 'object':
    df['label'] = df['label'].astype('category').cat.codes

if df['label'].min() > 0:
    min_label = df['label'].min()
    df['label'] = df['label'] - min_label

num_labels = df['label'].nunique()
print(f"\nКількість унікальних класів (num_labels): {num_labels}")

stratify_data = df['label'] if len(df) >= num_labels else None

train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['text'].tolist(),
    df['label'].tolist(),
    test_size=0.2,
    random_state=42,
    stratify=stratify_data
)

MODEL_NAME = 'roberta-base'
MAX_SEQ_LENGTH = 64

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

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)
model.to(DEVICE) 

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,             
    per_device_train_batch_size=32, 
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    eval_strategy='epoch',
    save_strategy='no',
    logging_dir='./logs',
    logging_steps=50,
    load_best_model_at_end=False,
    report_to="none",
    fp16=(DEVICE.type == 'cuda')
)

#Метрики
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    average_mode = 'weighted'
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average=average_mode, zero_division=0)
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'precision': precision, 'recall': recall, 'f1': f1}

print(f"\nПОЧАТОК ТРЕНУВАННЯ НА МОДЕЛІ {MODEL_NAME}")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

print(f"\n=== ЗБЕРЕЖЕННЯ МОДЕЛІ У {MODEL_SAVE_PATH} ===")
if not os.path.exists(MODEL_SAVE_PATH):
    os.makedirs(MODEL_SAVE_PATH)

model.save_pretrained(MODEL_SAVE_PATH)
tokenizer.save_pretrained(MODEL_SAVE_PATH)
print("Модель та токенізатор УСПІШНО збережено.")

print("\nОЦІНКА МОДЕЛІ ПІСЛЯ НАВЧАННЯ")
results = trainer.evaluate()
print("\nМетрики оцінки:")
for k, v in results.items():
    print(f"{k}: {v:.4f}")