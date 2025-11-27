import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Налаштування стилю графіків
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'DejaVu Sans' # Підтримка кирилиці

# --- ДАНІ (Взяті з вашого звіту) ---

# Назви класів (українською для звіту)
CLASS_NAMES = [
    "Нейтральна\nпідтримка (0)",
    "Активна\nпідтримка (1)",
    "Порада (2)",
    "Відсутність\nемпатії (3)",
    "Підбадьорення (4)",
    "Інше (5)"
]

# Дані метрик (F1-score)
f1_scores = [0.91, 0.84, 0.64, 0.92, 0.00, 0.95]

# Матриця плутанини (з вашого логу)
cm_data = np.array([
    [837,  21,   4,   3,   0,  28],
    [ 27, 125,   0,   0,   0,   0],
    [  2,   0,  54,  47,   0,   0],
    [  3,   0,   6, 367,   0,   0],
    [ 51,   0,   0,   0,   0,  63],
    [ 33,   0,   1,   4,   0, 1297]
])

def plot_metrics_bar():
    """Малює стовпчикову діаграму F1-score (Рисунок 3.1)"""
    plt.figure(figsize=(10, 6))
    
    # Створення барів
    colors = ['#4c72b0' if x > 0 else '#c44e52' for x in f1_scores] # Червоний для нульового класу
    bars = plt.bar(CLASS_NAMES, f1_scores, color=colors, alpha=0.8)
    
    # Додавання значень над стовпчиками
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.2f}',
                 ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.title('Ефективність класифікації за класами (F1-score)', fontsize=14, pad=20)
    plt.xlabel('Класи емпатії', fontsize=12)
    plt.ylabel('Значення F1-score', fontsize=12)
    plt.ylim(0, 1.1) # Трохи більше 1 для місця під текст
    
    plt.tight_layout()
    filename = 'metric_barchart.png'
    plt.savefig(filename, dpi=300)
    print(f"Графік метрик збережено у файл: {filename}")
    plt.close()

def plot_confusion_matrix():
    """Малює теплову карту матриці плутанини"""
    plt.figure(figsize=(10, 8))
    
    # Створення теплової карти
    sns.heatmap(cm_data, annot=True, fmt='d', cmap='Blues', 
                xticklabels=CLASS_NAMES, 
                yticklabels=CLASS_NAMES,
                linewidths=.5, linecolor='gray', cbar=False)
    
    plt.title('Матриця плутанини (Confusion Matrix)', fontsize=14, pad=20)
    plt.ylabel('Істинний клас', fontsize=12)
    plt.xlabel('Передбачений клас', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    filename = 'confusion_matrix.png'
    plt.savefig(filename, dpi=300)
    print(f"Матрицю плутанини збережено у файл: {filename}")
    plt.close()

if __name__ == "__main__":
    print("Генерація графіків...")
    plot_metrics_bar()
    plot_confusion_matrix()
    print("Готово!")