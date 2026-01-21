from flask import Flask, request, jsonify, render_template, redirect, url_for, session, flash
import pandas as pd
from collections import Counter
import joblib
import logging
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
import os
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
import ollama
import tempfile
import shutil

# Set up cache directory for Hugging Face
cache_dir = os.path.join(tempfile.gettempdir(), 'hf_cache')
os.makedirs(cache_dir, exist_ok=True)
os.environ['HF_HOME'] = cache_dir
os.environ['TRANSFORMERS_CACHE'] = cache_dir
os.environ['SENTENCE_TRANSFORMERS_HOME'] = cache_dir

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Change this to a secure secret key
app.config['SESSION_TYPE'] = 'filesystem'

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Database initialization
def init_db():
    db_path = os.path.join(tempfile.gettempdir(), 'users.db')
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (email TEXT PRIMARY KEY, password TEXT)''')
    
    # Create prediction history table
    c.execute('''CREATE TABLE IF NOT EXISTS prediction_history
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_email TEXT,
                  symptoms TEXT,
                  predicted_disease TEXT,
                  confidence FLOAT,
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY (user_email) REFERENCES users(email))''')
    
    # Create quiz history table
    c.execute('''CREATE TABLE IF NOT EXISTS quiz_history
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_email TEXT,
                  quiz_type TEXT,
                  score INTEGER,
                  result TEXT,
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY (user_email) REFERENCES users(email))''')
    conn.commit()
    conn.close()
    return db_path

# Initialize database
db_path = init_db()

try:
    # Load models and data
    logging.info("Loading datasets and models...")
    dis_sym_data_v1 = pd.read_csv("Processed_Dataset1.csv")
    doc_data = pd.read_csv("Doctor_Versus_Disease.csv", encoding='latin1', names=['Disease', 'Specialist'])
    des_data = pd.read_csv("Disease_Description.csv")
    algorithms = joblib.load(r"trained_algorithms.pkl")
    le = joblib.load("label_encoder.pkl")
    
    # Load chatbot data and model
    chatbot_df = pd.read_csv('chatbot final ques dataset.csv',encoding='ISO-8859-1')
    chatbot_df = chatbot_df.dropna(subset=['User Question'])
    # Fix the pandas warning
    chatbot_df = chatbot_df.copy()
    chatbot_df.loc[:, 'User Question'] = chatbot_df['User Question'].fillna('')
    
    # Load sentence transformer model with cache directory
    sentence_model = SentenceTransformer('paraphrase-MiniLM-L6-v2', cache_folder=cache_dir)
    
    # Enable tqdm progress bar
    tqdm.pandas()
    
    # Encode questions
    chatbot_df['question_embedding'] = chatbot_df['User Question'].progress_apply(
        lambda x: sentence_model.encode(str(x), convert_to_tensor=True)
    )
    chatbot_df["Answer"] = chatbot_df["AI Response"]
    
    # Clean dataset
    dis_sym_data_v1 = dis_sym_data_v1.loc[:, ~dis_sym_data_v1.columns.str.contains('^Unnamed')]
    test_col = [col for col in dis_sym_data_v1.columns if col != 'Disease']
    logging.info("Datasets and models loaded successfully!")
except Exception as e:
    logging.error(f"Error loading models or datasets: {e}")
    exit(1)

# Login required decorator
def login_required(f):
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            flash('Please login to access this page.', 'error')
            return redirect(url_for('home'))
        return f(*args, **kwargs)
    decorated_function.__name__ = f.__name__
    return decorated_function

@app.route('/')
def home():
    if 'user' in session:
        return redirect(url_for('predict_page'))
    return render_template('landing.html')

@app.route('/signup')
def signup():
    if 'user' in session:
        return redirect(url_for('predict_page'))
    return render_template('signup.html')

@app.route('/signup', methods=['POST'])
def signup_post():
    email = request.form.get('email')
    password = request.form.get('password')
    confirm_password = request.form.get('confirm_password')

    if not email or not password:
        flash('Please fill in all fields.', 'error')
        return redirect(url_for('signup'))

    if password != confirm_password:
        flash('Passwords do not match.', 'error')
        return redirect(url_for('signup'))

    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    # Check if user already exists
    c.execute('SELECT email FROM users WHERE email = ?', (email,))
    if c.fetchone():
        flash('Email already registered.', 'error')
        conn.close()
        return redirect(url_for('signup'))

    # Hash password and store user
    hashed_password = generate_password_hash(password)
    c.execute('INSERT INTO users (email, password) VALUES (?, ?)',
              (email, hashed_password))
    conn.commit()
    conn.close()

    flash('Registration successful! Please login.', 'success')
    return redirect(url_for('login_page'))

@app.route('/login')
def login_page():
    if 'user' in session:
        return redirect(url_for('predict_page'))
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login():
    email = request.form.get('email')
    password = request.form.get('password')

    if not email or not password:
        flash('Please fill in all fields.', 'error')
        return redirect(url_for('login_page'))

    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('SELECT password FROM users WHERE email = ?', (email,))
    result = c.fetchone()
    conn.close()

    if result and check_password_hash(result[0], password):
        session['user'] = email
        flash('Login successful!', 'success')
        return redirect(url_for('predict_page'))
    else:
        flash('Invalid email or password.', 'error')
        return redirect(url_for('login_page'))

@app.route('/logout')
def logout():
    session.pop('user', None)
    flash('You have been logged out.', 'success')
    return redirect(url_for('home'))

@app.route('/predict')
@login_required
def predict_page():
    # Get all symptoms from the dataset
    symptoms = test_col  # This already contains all symptoms from the dataset
    return render_template('prediction.html', symptoms=symptoms)

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    try:
        data = request.get_json()
        logging.debug(f"Received Data: {data}")

        symptoms = data.get('symptoms', [])
        if not symptoms:
            return jsonify({"error": "No symptoms provided."}), 400

        # Format symptoms to match dataset column names
        formatted_symptoms = []
        for symptom in symptoms:
            formatted_symptom = symptom.lower().replace(' ', '_')
            formatted_symptom = ''.join(c for c in formatted_symptom if c.isalnum() or c == '_')
            formatted_symptoms.append(formatted_symptom)

        # Create test data with formatted symptom names
        test_data = {col: 1 if col in formatted_symptoms else 0 for col in test_col}
        test_df = pd.DataFrame([test_data])
        logging.debug(f"Test DataFrame: {test_df}")

        predicted = []
        for model_name, values in algorithms.items():
            predict_disease = values["model"].predict(test_df)
            predict_disease = le.inverse_transform(predict_disease)
            predicted.extend(predict_disease)

        disease_counts = Counter(predicted)
        percentage_per_disease = {disease: (count / len(algorithms)) * 100 for disease, count in disease_counts.items()}

        result_df = pd.DataFrame({"Disease": list(percentage_per_disease.keys()),
                                  "Chances": list(percentage_per_disease.values())})
        result_df = result_df.merge(doc_data, on='Disease', how='left')
        result_df = result_df.merge(des_data, on='Disease', how='left')

        # Store prediction history
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        
        # Store the top prediction
        top_prediction = result_df.iloc[0]
        c.execute('''INSERT INTO prediction_history 
                    (user_email, symptoms, predicted_disease, confidence)
                    VALUES (?, ?, ?, ?)''',
                 (session['user'],
                  ','.join(symptoms),
                  top_prediction['Disease'],
                  top_prediction['Chances']))
        conn.commit()
        conn.close()

        results = []
        for _, row in result_df.iterrows():
            results.append({
                "Disease": row['Disease'],
                "Chances": row['Chances'],
                "Specialist": row.get('Specialist', 'Specialist info not available'),
                "Description": row.get('Description', 'Description not available')
            })

        logging.debug(f"Final Results: {results}")
        return jsonify(results)
    except Exception as e:
        logging.error(f"Error in prediction: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/quiz/heart')
@login_required
def heart_quiz():
    return render_template('quiz_heart.html')

@app.route('/quiz/lung')
@login_required
def lung_quiz():
    return render_template('quiz_lung.html')

@app.route('/quiz/heart', methods=['POST'])
@login_required
def heart_quiz_submit():
    try:
        data = request.get_json()
        score = data.get('score', 0)
        result = data.get('result', '')
        
        # Store quiz result
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute('''INSERT INTO quiz_history 
                    (user_email, quiz_type, score, result)
                    VALUES (?, ?, ?, ?)''',
                 (session['user'], 'Heart Health Quiz', score, result))
        conn.commit()
        conn.close()
        
        return jsonify({"success": True, "message": "Quiz result saved successfully"})
    except Exception as e:
        logging.error(f"Error saving heart quiz result: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/quiz/lung', methods=['POST'])
@login_required
def lung_quiz_submit():
    try:
        data = request.get_json()
        score = data.get('score', 0)
        result = data.get('result', '')
        
        # Store quiz result
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute('''INSERT INTO quiz_history 
                    (user_email, quiz_type, score, result)
                    VALUES (?, ?, ?, ?)''',
                 (session['user'], 'Lung Health Quiz', score, result))
        conn.commit()
        conn.close()
        
        return jsonify({"success": True, "message": "Quiz result saved successfully"})
    except Exception as e:
        logging.error(f"Error saving lung quiz result: {e}")
        return jsonify({"error": str(e)}), 500

# Chatbot routes
@app.route('/chatbot')
@login_required
def chatbot():
    if 'chat_history' not in session:
        session['chat_history'] = []
    return render_template('chatbot.html', chat_history=session['chat_history'])

def find_most_similar_answer(user_question):
    # Encode user's question
    user_question_embedding = sentence_model.encode(user_question, convert_to_tensor=True)

    # Compute similarities directly with embeddings
    similarities = [util.pytorch_cos_sim(user_question_embedding, emb).item() for emb in chatbot_df['question_embedding']]

    # Find the most similar question
    most_similar_idx = similarities.index(max(similarities))
    most_similar_row = chatbot_df.iloc[most_similar_idx]

    return most_similar_row['User Question'], most_similar_row['Answer']

def generate_ollama_response(chat_history, relevant_question, relevant_answer):
    # For Hugging Face deployment, we'll use a simpler response without Ollama
    # since Ollama isn't available in Hugging Face Spaces
    prompt = f"Based on the following information: \"{relevant_answer}\", provide a professional response that addresses the user's query directly and concisely."
    
    # Return a simple enhanced response
    return f"Based on the medical information available: {relevant_answer}\n\nThis response is based on our healthcare knowledge base. For personalized medical advice, please consult with a healthcare professional."

@app.route('/chatbot/ask', methods=['POST'])
@login_required
def chatbot_ask():
    try:
        data = request.get_json()
        question = data.get('question')
        
        if not question:
            return jsonify({"error": "No question provided"}), 400

        # Initialize chat history if not exists
        if 'chat_history' not in session:
            session['chat_history'] = []

        # Find the most similar question and its answer from the dataset
        relevant_question, relevant_answer = find_most_similar_answer(question)
        
        # Add this exchange to the chat history
        session['chat_history'].append({"user": question, "bot": relevant_answer})
        
        # Generate a more contextual response
        enhanced_answer = generate_ollama_response(session['chat_history'], relevant_question, relevant_answer)
        
        # Update the last bot response with the enhanced answer
        session['chat_history'][-1]['bot'] = enhanced_answer
        
        # Save the updated chat history to the session
        session.modified = True
        
        return jsonify({"response": enhanced_answer})
    except Exception as e:
        logging.error(f"Error in chatbot: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/profile')
@login_required
def profile():
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    # Get user's prediction history
    c.execute('''SELECT predicted_disease, confidence, timestamp, symptoms 
                 FROM prediction_history 
                 WHERE user_email = ? 
                 ORDER BY timestamp DESC''', (session['user'],))
    predictions = c.fetchall()
    
    # Get prediction statistics
    c.execute('''SELECT predicted_disease, COUNT(*) as count, AVG(confidence) as avg_confidence
                 FROM prediction_history 
                 WHERE user_email = ? 
                 GROUP BY predicted_disease''', (session['user'],))
    stats = c.fetchall()
    
    # Get quiz history
    c.execute('''SELECT quiz_type, score, result, timestamp 
                 FROM quiz_history 
                 WHERE user_email = ? 
                 ORDER BY timestamp DESC''', (session['user'],))
    quiz_history = c.fetchall()
    
    # Get quiz statistics
    c.execute('''SELECT quiz_type, COUNT(*) as count, AVG(score) as avg_score
                 FROM quiz_history 
                 WHERE user_email = ? 
                 GROUP BY quiz_type''', (session['user'],))
    quiz_stats = c.fetchall()
    
    conn.close()
    
    return render_template('profile.html', 
                         predictions=predictions,
                         stats=stats,
                         quiz_history=quiz_history,
                         quiz_stats=quiz_stats,
                         user_email=session['user'])

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=7860) 