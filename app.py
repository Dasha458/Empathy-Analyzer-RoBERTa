import os
import matplotlib
matplotlib.use('Agg') 
import torch
from flask import Flask, request, render_template, redirect, url_for, session
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timedelta
import traceback
import pandas as pd
import matplotlib.pyplot as plt
import base64
from io import BytesIO

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///empathy_logs.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.secret_key = 'super_secret_key_123'
db = SQLAlchemy(app)

class EmpathyLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    role = db.Column(db.String(20), nullable=False)
    person_name = db.Column(db.String(100), nullable=True)
    text_input = db.Column(db.Text, nullable=False)
    predicted_label = db.Column(db.String(100), nullable=False)
    predicted_score = db.Column(db.Integer, nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

with app.app_context():
    db.create_all()

MODEL_DIR = './model_files/'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CLASS_MAPPING = {
    0: "Нейтральна підтримка (Passive Support)",
    1: "Активна підтримка (Active Empathy)",
    2: "Надання поради/інформації (Advice)",
    3: "Відсутність емпатії/Критика (Lack of Empathy)",
    4: "Підбадьорення/Позитив (Encouragement)",
    5: "Інше/Недоречний коментар (Other)"
}

CLASS_TO_SCORE = {
    "Активна підтримка (Active Empathy)": 5,
    "Підбадьорення/Позитив (Encouragement)": 4,
    "Нейтральна підтримка (Passive Support)": 3,
    "Надання поради/інформації (Advice)": 2,
    "Відсутність емпатії/Критика (Lack of Empathy)": 1,
    "Інше/Недоречний коментар (Other)": 1
}

tokenizer = None
model = None
MAX_SEQ_LENGTH = 64

try:
    print(f"Завантаження моделі з {MODEL_DIR} на пристрій: {DEVICE}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.to(DEVICE)
    model.eval()
    print("Модель успішно завантажена.")
except Exception as e:
    print("-" * 50)
    print("КРИТИЧНА ПОМИЛКА ЗАВАНТАЖЕННЯ МОДЕЛІ")
    print(f"Помилка: {e}")
    traceback.print_exc()
    print("-" * 50)

def predict_empathy(text):
    if tokenizer is None or model is None:
        return "Помилка: Модель не завантажена.", 0.0, 0
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=MAX_SEQ_LENGTH)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    probabilities = torch.softmax(outputs.logits, dim=1)[0]
    predicted_index = torch.argmax(probabilities).item()
    predicted_label = CLASS_MAPPING.get(predicted_index, f"Невідомий клас ({predicted_index})")
    confidence = probabilities[predicted_index].item() * 100
    
    predicted_score = CLASS_TO_SCORE.get(predicted_label, 1)
    
    return predicted_label, confidence, predicted_score

@app.route('/', methods=['GET'])
def home():
    return redirect(url_for('select_role'))

@app.route('/select_role', methods=['GET', 'POST'])
def select_role():
    if request.method == 'POST':
        selected_role = request.form.get('role')
        session['role'] = selected_role
        return redirect(url_for('index'))
    return render_template('select_role.html')

@app.route('/index', methods=['GET'])
def index():
    role = session.get('role', 'user')
    return render_template('index.html', role=role)

@app.route('/predict', methods=['POST'])
def predict():
    role = session.get('role', 'user')
    user_input = request.form.get('text_input', '')
    person_name = request.form.get('person_name') if role == 'psychologist' else None

    if not user_input.strip():
        return render_template('index.html', result="Будь ласка, введіть текст для аналізу.", role=role)

    label, confidence, score = predict_empathy(user_input)
    
    if score >= 4:
        result_class = 'result-high'
    elif score == 3:
        result_class = 'result-medium'
    else: # 1, 2
        result_class = 'result-low'

    if "Помилка" not in label:
        new_log = EmpathyLog(
            role=role,
            person_name=person_name,
            text_input=user_input,
            predicted_label=label,
            predicted_score=score,
            confidence=confidence
        )
        db.session.add(new_log)
        try:
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            print(f"Помилка при збереженні: {e}")
            return render_template('index.html', result="Сталася внутрішня помилка. Спробуйте, будь ласка, пізніше.", role=role)

        result_text = f"""
            <p><strong>Результат аналізу:</strong></p>
            <p><strong>Клас емпатії:</strong> {label}</p>
            <p><strong>Оцінка:</strong> {score}/5</p>
            <p class="score-explain">
                <small>Оцінка (1-5) відображає силу емпатії відповіді:
                5 — Активна емпатія/найвища підтримка, 1 — Критика/відсутність емпатії.</small>
            </p>
            <p><strong>Впевненість моделі:</strong> {confidence:.2f}%</p>
            <p class="confidence-explain">
                <small>Впевненість показує, наскільки нейромережа впевнена у своєму виборі класу. 
                Чим вищий відсоток, тим надійніша класифікація.</small>
            </p>
        """
    else:
        result_text = label
        result_class = 'result-low'


    return render_template('index.html', 
                           result=result_text, 
                           result_class=result_class, 
                           original_text=user_input, 
                           role=role)

@app.route('/history', methods=['GET', 'POST'])
def history():
    role_filter = session.get('role', request.form.get('role', 'user'))
    
    person_name_filter = request.values.get('person_name', '').strip()

    query = EmpathyLog.query.filter(EmpathyLog.role == role_filter)
    
    if person_name_filter and role_filter == 'psychologist':
        query = query.filter(EmpathyLog.person_name == person_name_filter) 

    logs = query.order_by(EmpathyLog.timestamp.desc()).all()

    if not logs:
        display_name = person_name_filter if person_name_filter else "невідоме"
        message = (
            f"Записи для імені '{display_name}' не знайдено."
            if person_name_filter else "Історія порожня."
        )
        return render_template('history.html', logs=[], message=message, role=role_filter, person_name_filter=person_name_filter)

    return render_template('history.html', logs=logs, message=None, role=role_filter, person_name_filter=person_name_filter)
@app.route('/graph', methods=['GET', 'POST'])
def graph():
    role = session.get('role', 'user')
    
    person_name_filter = request.form.get('person_name', '').strip()
    time_range = request.form.get('time_range', 'week')
    
    end_date = datetime.utcnow()

    if time_range == 'day':
        start_date = end_date - timedelta(hours=24)
        freq = None 
    elif time_range == 'week':
        start_date = end_date - timedelta(weeks=1)
        freq = 'D'
    elif time_range == 'month':
        start_date = end_date - timedelta(days=30)
        freq = 'D'
    else:
        start_date = end_date - timedelta(weeks=1)
        freq = 'D'

    unique_names_query = db.session.query(EmpathyLog.person_name).distinct().filter(
        EmpathyLog.person_name != None, EmpathyLog.person_name != ''
    ).order_by(EmpathyLog.person_name).all()
    unique_names = [name[0] for name in unique_names_query]

    query = EmpathyLog.query.filter(EmpathyLog.timestamp >= start_date)
    
    if person_name_filter:
        query = query.filter(EmpathyLog.person_name == person_name_filter)
        
    recent_logs = query.order_by(EmpathyLog.timestamp.asc()).all()
    data = [{'timestamp': log.timestamp, 'score': log.predicted_score} for log in recent_logs]

    if not data:
        return render_template('graph.html', graph_image=None, time_range=time_range, role=role, person_name_filter=person_name_filter, unique_names=unique_names)

    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)

    if freq:
        trend_data = df['score'].resample(freq).mean().dropna()
    else:
        trend_data = df['score']
    
    if trend_data.empty:
        return render_template('graph.html', graph_image=None, time_range=time_range, role=role, person_name_filter=person_name_filter, unique_names=unique_names)

    plt.figure(figsize=(10, 6))
    
    if freq:
        plt.plot(trend_data.index, trend_data.values, marker='o', linestyle='-', color='#007bff')
    else:
        plt.plot(trend_data.index, trend_data.values, 'o', linestyle='-', color='#007bff')
    
    title_suffix = f" для {person_name_filter}" if person_name_filter else ""
    plt.title(f'Тренд Рівня Емпатії ({time_range.capitalize()}){title_suffix}')
    plt.xlabel('Дата/Час')
    plt.ylabel('Середній Рівень Емпатії (1 - 5)')
    
    plt.ylim(0.8, 5.2) 
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45, ha='right')
    
    if time_range == 'day' or (freq is None):
        plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))
    elif time_range == 'week':
        plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%a, %b %d'))
    else: # month
        plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b %d'))


    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    graph_image = base64.b64encode(buf.getvalue()).decode('utf-8')

    return render_template('graph.html', graph_image=graph_image, time_range=time_range, role=role, person_name_filter=person_name_filter, unique_names=unique_names)

if __name__ == '__main__':
    if not os.path.exists('templates'):
        os.makedirs('templates')
    app.run(debug=True)
