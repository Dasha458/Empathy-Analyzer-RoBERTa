import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'DejaVu Sans' 

CLASS_NAMES = [
    "Нейтральна\nпідтримка (0)",
    "Активна\nпідтримка (1)",
    "Порада (2)",
    "Відсутність\nемпатії (3)",
    "Підбадьорення (4)",
    "Інше (5)"
]

f1_scores = [0.99, 0.99, 1.00, 1.00, 0.97, 1.00]

cm_data = np.array([
    [886,   2,   0,   0,   3,   2],
    [  0, 152,   0,   0,   0,   0],
    [  0,   0, 103,   0,   0,   0],
    [  0,   0,   1, 375,   0,   0],
    [  2,   0,   0,   0, 112,   0],
    [  2,   0,   0,   1,   1, 1331]
])

def plot_metrics_bar():
    """Малює стовпчикову діаграму F1-score (Рисунок 3.1)"""
    plt.figure(figsize=(10, 6))
    
    colors = ['#4c72b0'] * len(f1_scores)
    
    bars = plt.bar(CLASS_NAMES, f1_scores, color=colors, alpha=0.85)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.2f}',
                 ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.title('Ефективність класифікації за класами (F1-score)', fontsize=14, pad=20)
    plt.xlabel('Класи емпатії', fontsize=12)
    plt.ylabel('Значення F1-score', fontsize=12)
    plt.ylim(0, 1.1) 
    
    plt.tight_layout()
    filename = 'metric_barchart_final.png'
    plt.savefig(filename, dpi=300)
    print(f"Графік метрик збережено у файл: {filename}")
    plt.close()

def plot_confusion_matrix():
    """Малює теплову карту матриці плутанини (Рисунок 3.2)"""
    plt.figure(figsize=(10, 8))
    
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
    filename = 'confusion_matrix_final.png'
    plt.savefig(filename, dpi=300)
    print(f"Матрицю плутанини збережено у файл: {filename}")
    plt.close()

if __name__ == "__main__":
    print("Генерація фінальних графіків...")
    plot_metrics_bar()
    plot_confusion_matrix()
    print("Готово! Перевірте папку проекту.")