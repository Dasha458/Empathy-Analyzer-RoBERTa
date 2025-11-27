import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from tqdm import tqdm

# --- КОНФІГУРАЦІЯ ---
MODEL_PATH = 'model_files/'   # Шлях до папки, де лежить навчена модель
DATASET_PATH = 'empathy.csv'  # Ваш файл з даними
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_SEQ_LENGTH = 64
BATCH_SIZE = 16

print(f"Використовується пристрій: {DEVICE}")

def load_data(path):
    """Завантаження даних під вашу структуру CSV."""
    try:
        df = pd.read_csv(path)
        
        # Очищення назв колонок від зайвих пробілів (на всяк випадок)
        df.columns = df.columns.str.strip()
        
        # Перевірка наявності необхідних колонок
        if 'empathetic_segments' not in df.columns or 'empathy_class' not in df.columns:
            raise ValueError(f"У файлі {path} не знайдено колонок 'empathetic_segments' або 'empathy_class'.\nНаявні колонки: {df.columns.tolist()}")

        # Перейменування для зручності
        # Ми беремо 'empathetic_segments' як текст для аналізу
        df = df.rename(columns={'empathetic_segments': 'text', 'empathy_class': 'label'})
        
        # Залишаємо тільки потрібні колонки і видаляємо пусті рядки
        df = df[['text', 'label']]
        df.dropna(subset=['text', 'label'], inplace=True)

        # Якщо мітки записані текстом (наприклад, "Active Empathy"), конвертуємо їх у числа.
        # Якщо вони вже числа (0, 1, 2...), цей крок нічого не зіпсує.
        if df['label'].dtype == 'object':
            df['label'] = df['label'].astype('category').cat.codes
        
        # Нормалізація міток, щоб вони починалися з 0
        # (Наприклад, якщо класи 1-6, робимо 0-5)
        if pd.api.types.is_numeric_dtype(df['label']) and df['label'].min() > 0:
            print(f"Зсув міток: віднімаємо {df['label'].min()}, щоб класи починалися з 0.")
            df['label'] = df['label'] - df['label'].min()
            
        print(f"Дані успішно завантажено. Кількість записів: {len(df)}")
        return df
        
    except Exception as e:
        print(f"Критична помилка при завантаженні даних: {e}")
        exit()

def evaluate():
    # 1. Завантаження даних
    print("--- Етап 1: Завантаження даних ---")
    df = load_data(DATASET_PATH)
    texts = df['text'].tolist()
    true_labels = df['label'].tolist()
    
    # 2. Завантаження моделі
    print("\n--- Етап 2: Завантаження моделі ---")
    try:
        print(f"Шукаємо модель у папці: {MODEL_PATH}")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        model.to(DEVICE)
        model.eval() # Важливо: режим оцінки
        print("Модель успішно завантажена!")
    except Exception as e:
        print(f"Не вдалося завантажити модель. Переконайтеся, що папка '{MODEL_PATH}' існує і містить файли.")
        print(f"Деталі помилки: {e}")
        exit()

    predicted_labels = []

    # 3. Передбачення
    print("\n--- Етап 3: Класифікація (Процес може зайняти час) ---")
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Обробка"):
            batch_texts = texts[i : i + BATCH_SIZE]
            
            # Токенізація
            inputs = tokenizer(
                batch_texts, 
                padding=True, 
                truncation=True, 
                max_length=MAX_SEQ_LENGTH, 
                return_tensors="pt"
            ).to(DEVICE)

            # Прогон через модель
            outputs = model(**inputs)
            
            # Вибір класу з найбільшою ймовірністю
            preds = torch.argmax(outputs.logits, dim=1)
            predicted_labels.extend(preds.cpu().numpy())

    # 4. Звіт
    print("\n" + "="*40)
    print(" РЕЗУЛЬТАТИ ПЕРЕВІРКИ")
    print("="*40)
    
    acc = accuracy_score(true_labels, predicted_labels)
    print(f"Точність (Accuracy): {acc:.2%}")
    print("-" * 40)
    
    print("Детальний звіт по класах:")
    # zero_division=0 прибирає попередження, якщо якогось класу немає у передбаченнях
    print(classification_report(true_labels, predicted_labels, zero_division=0))
    
    print("-" * 40)
    print("Матриця плутанини (Confusion Matrix):")
    print(confusion_matrix(true_labels, predicted_labels))

if __name__ == "__main__":
    evaluate()