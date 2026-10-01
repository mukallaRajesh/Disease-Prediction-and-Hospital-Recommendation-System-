import os
from pathlib import Path
from flask import Flask, request, jsonify, render_template, redirect, url_for, session, flash
import pandas as pd
from collections import Counter
import joblib
import logging
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
from openai import OpenAI

BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / ".hf_cache"
CACHE_DIR.mkdir(exist_ok=True)

os.environ["HF_HOME"] = str(CACHE_DIR)
os.environ["TRANSFORMERS_CACHE"] = str(CACHE_DIR)
os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(CACHE_DIR)

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "your_secret_key")
app.config["SESSION_TYPE"] = "filesystem"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.6-luna")
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

logging.basicConfig(level=logging.INFO)

def init_db():
    db_path = BASE_DIR / "users.db"
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    c.execute(
        """CREATE TABLE IF NOT EXISTS users
        (email TEXT PRIMARY KEY, password TEXT)"""
    )

    c.execute(
        """CREATE TABLE IF NOT EXISTS prediction_history
        (id INTEGER PRIMARY KEY AUTOINCREMENT,
         user_email TEXT,
         symptoms TEXT,
         predicted_disease TEXT,
         confidence FLOAT,
         timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
         FOREIGN KEY (user_email) REFERENCES users(email))"""
    )

    c.execute(
        """CREATE TABLE IF NOT EXISTS quiz_history
        (id INTEGER PRIMARY KEY AUTOINCREMENT,
         user_email TEXT,
         quiz_type TEXT,
         score INTEGER,
         result TEXT,
         timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
         FOREIGN KEY (user_email) REFERENCES users(email))"""
    )

    conn.commit()
    conn.close()

init_db()

try:
    logging.info("Loading datasets and models...")

    dis_sym_data_v1 = pd.read_csv(BASE_DIR / "Processed_Dataset1.csv")
    doc_data = pd.read_csv(
        BASE_DIR / "Doctor_Versus_Disease.csv",
        encoding="latin1",
        names=["Disease", "Specialist"]
    )
    des_data = pd.read_csv(BASE_DIR / "Disease_Description.csv")
    algorithms = joblib.load(BASE_DIR / "trained_algorithms.pkl")
    le = joblib.load(BASE_DIR / "label_encoder.pkl")
    chatbot_df = pd.read_csv(BASE_DIR / "chatbot.csv", encoding="latin1")
    chatbot_df["User Question"] = chatbot_df["User Question"].fillna("")

    sentence_model = SentenceTransformer(
        "all-MiniLM-L6-v2",
        cache_folder=str(CACHE_DIR)
    )

    tqdm.pandas()

    chatbot_df["question_embedding"] = chatbot_df["User Question"].progress_apply(
        lambda x: sentence_model.encode(str(x), convert_to_tensor=True)
    )

    chatbot_df["Answer"] = chatbot_df["AI Response"]

    dis_sym_data_v1 = dis_sym_data_v1.loc[
        :,
        ~dis_sym_data_v1.columns.str.contains("^Unnamed")
    ]

    test_col = [col for col in dis_sym_data_v1.columns if col != "Disease"]

    logging.info("Datasets and models loaded successfully!")

except Exception as e:
    logging.error(f"Error loading models or datasets: {e}")
    raise

def login_required(f):
    def decorated_function(*args, **kwargs):
        if "user" not in session:
            flash("Please login to access this page.", "error")
            return redirect(url_for("home"))
        return f(*args, **kwargs)

    decorated_function.__name__ = f.__name__
    return decorated_function

@app.route("/")
def home():
    if "user" in session:
        return redirect(url_for("predict_page"))
    return render_template("landing.html")

@app.route("/signup")
def signup():
    if "user" in session:
        return redirect(url_for("predict_page"))
    return render_template("signup.html")

@app.route("/signup", methods=["POST"])
def signup_post():
    email = request.form.get("email")
    password = request.form.get("password")
    confirm_password = request.form.get("confirm_password")

    if not email or not password:
        flash("Please fill in all fields.", "error")
        return redirect(url_for("signup"))

    if password != confirm_password:
        flash("Passwords do not match.", "error")
        return redirect(url_for("signup"))

    conn = sqlite3.connect(BASE_DIR / "users.db")
    c = conn.cursor()

    c.execute("SELECT email FROM users WHERE email = ?", (email,))

    if c.fetchone():
        flash("Email already registered.", "error")
        conn.close()
        return redirect(url_for("signup"))

    hashed_password = generate_password_hash(password)

    c.execute(
        "INSERT INTO users (email, password) VALUES (?, ?)",
        (email, hashed_password)
    )

    conn.commit()
    conn.close()

    flash("Registration successful! Please login.", "success")
    return redirect(url_for("login_page"))

@app.route("/login")
def login_page():
    if "user" in session:
        return redirect(url_for("predict_page"))
    return render_template("login.html")

@app.route("/login", methods=["POST"])
def login():
    email = request.form.get("email")
    password = request.form.get("password")

    if not email or not password:
        flash("Please fill in all fields.", "error")
        return redirect(url_for("login_page"))

    conn = sqlite3.connect(BASE_DIR / "users.db")
    c = conn.cursor()

    c.execute("SELECT password FROM users WHERE email = ?", (email,))
    result = c.fetchone()
    conn.close()

    if result and check_password_hash(result[0], password):
        session["user"] = email
        flash("Login successful!", "success")
        return redirect(url_for("predict_page"))

    flash("Invalid email or password.", "error")
    return redirect(url_for("login_page"))

@app.route("/logout")
def logout():
    session.pop("user", None)
    flash("You have been logged out.", "success")
    return redirect(url_for("home"))

@app.route("/predict")
@login_required
def predict_page():
    symptoms = test_col
    return render_template("prediction.html", symptoms=symptoms)

@app.route("/predict", methods=["POST"])
@login_required
def predict():
    try:
        data = request.get_json()
        logging.debug(f"Received Data: {data}")

        symptoms = data.get("symptoms", [])

        if not symptoms:
            return jsonify({"error": "No symptoms provided."}), 400

        formatted_symptoms = []

        for symptom in symptoms:
            formatted_symptom = symptom.lower().replace(" ", "_")
            formatted_symptom = "".join(
                c for c in formatted_symptom
                if c.isalnum() or c == "_"
            )
            formatted_symptoms.append(formatted_symptom)

        test_data = {
            col: 1 if col in formatted_symptoms else 0
            for col in test_col
        }

        test_df = pd.DataFrame([test_data])

        predicted = []

        for model_name, values in algorithms.items():
            predict_disease = values["model"].predict(test_df)
            predict_disease = le.inverse_transform(predict_disease)
            predicted.extend(predict_disease)

        disease_counts = Counter(predicted)

        percentage_per_disease = {
            disease: (count / len(algorithms)) * 100
            for disease, count in disease_counts.items()
        }

        result_df = pd.DataFrame({
            "Disease": list(percentage_per_disease.keys()),
            "Chances": list(percentage_per_disease.values())
        })

        result_df = result_df.merge(
            doc_data,
            on="Disease",
            how="left"
        )

        result_df = result_df.merge(
            des_data,
            on="Disease",
            how="left"
        )

        conn = sqlite3.connect(BASE_DIR / "users.db")
        c = conn.cursor()

        top_prediction = result_df.iloc[0]

        c.execute(
            """INSERT INTO prediction_history
            (user_email, symptoms, predicted_disease, confidence)
            VALUES (?, ?, ?, ?)""",
            (
                session["user"],
                ",".join(symptoms),
                top_prediction["Disease"],
                top_prediction["Chances"]
            )
        )

        conn.commit()
        conn.close()

        results = []

        for _, row in result_df.iterrows():
            results.append({
                "Disease": row["Disease"],
                "Chances": row["Chances"],
                "Specialist": row.get(
                    "Specialist",
                    "Specialist info not available"
                ),
                "Description": row.get(
                    "Description",
                    "Description not available"
                )
            })

        return jsonify(results)

    except Exception as e:
        logging.error(f"Error in prediction: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/quiz/heart")
@login_required
def heart_quiz():
    return render_template("quiz_heart.html")

@app.route("/quiz/lung")
@login_required
def lung_quiz():
    return render_template("quiz_lung.html")

@app.route("/quiz/heart", methods=["POST"])
@login_required
def heart_quiz_submit():
    try:
        data = request.get_json()
        score = data.get("score", 0)
        result = data.get("result", "")

        conn = sqlite3.connect(BASE_DIR / "users.db")
        c = conn.cursor()

        c.execute(
            """INSERT INTO quiz_history
            (user_email, quiz_type, score, result)
            VALUES (?, ?, ?, ?)""",
            (
                session["user"],
                "Heart Health Quiz",
                score,
                result
            )
        )

        conn.commit()
        conn.close()

        return jsonify({
            "success": True,
            "message": "Quiz result saved successfully"
        })

    except Exception as e:
        logging.error(f"Error saving heart quiz result: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/quiz/lung", methods=["POST"])
@login_required
def lung_quiz_submit():
    try:
        data = request.get_json()
        score = data.get("score", 0)
        result = data.get("result", "")

        conn = sqlite3.connect(BASE_DIR / "users.db")
        c = conn.cursor()

        c.execute(
            """INSERT INTO quiz_history
            (user_email, quiz_type, score, result)
            VALUES (?, ?, ?, ?)""",
            (
                session["user"],
                "Lung Health Quiz",
                score,
                result
            )
        )

        conn.commit()
        conn.close()

        return jsonify({
            "success": True,
            "message": "Quiz result saved successfully"
        })

    except Exception as e:
        logging.error(f"Error saving lung quiz result: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/chatbot")
@login_required
def chatbot():
    if "chat_history" not in session:
        session["chat_history"] = []

    return render_template(
        "chatbot.html",
        chat_history=session["chat_history"]
    )

def find_most_similar_answer(user_question):
    user_question_embedding = sentence_model.encode(
        user_question,
        convert_to_tensor=True
    )

    similarities = [
        util.pytorch_cos_sim(
            user_question_embedding,
            emb
        ).item()
        for emb in chatbot_df["question_embedding"]
    ]

    most_similar_idx = similarities.index(max(similarities))
    most_similar_row = chatbot_df.iloc[most_similar_idx]

    return (
        most_similar_row["User Question"],
        most_similar_row["Answer"]
    )

def generate_ollama_response(
    chat_history,
    relevant_question,
    relevant_answer
):
    prompt = (
        "This is a conversation between a user and a health chatbot "
        "(in 2-3 points).\n"
    )

    for entry in chat_history:
        prompt += (
            f"User: {entry['user']}\n"
            f"Bot: {entry['bot']}\n"
        )

    prompt += (
        f"User: {chat_history[-1]['user']}\n"
        f"Based on the following information: \"{relevant_answer}\", "
        "provide a professional response that addresses the user's query "
        "directly and concisely without unnecessary introductory remarks "
        "and formatting styles and reply like a professional chatbot. "
        "Mention reference URL if any.\n"
        "Bot:"
    )

    if openai_client is None:
        raise RuntimeError("OPENAI_API_KEY is not configured.")

    response = openai_client.responses.create(
        model=OPENAI_MODEL,
        input=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.output_text

@app.route("/chatbot/ask", methods=["POST"])
@login_required
def chatbot_ask():
    try:
        data = request.get_json()
        question = data.get("question")

        if not question:
            return jsonify({"error": "No question provided"}), 400

        if "chat_history" not in session:
            session["chat_history"] = []

        relevant_question, relevant_answer = find_most_similar_answer(question)

        session["chat_history"].append({
            "user": question,
            "bot": relevant_answer
        })

        enhanced_answer = generate_ollama_response(
            session["chat_history"],
            relevant_question,
            relevant_answer
        )

        session["chat_history"][-1]["bot"] = enhanced_answer
        session.modified = True

        return jsonify({"response": enhanced_answer})

    except Exception as e:
        logging.error(f"Error in chatbot: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/profile")
@login_required
def profile():
    conn = sqlite3.connect(BASE_DIR / "users.db")
    c = conn.cursor()

    c.execute(
        """SELECT predicted_disease, confidence, timestamp, symptoms
        FROM prediction_history
        WHERE user_email = ?
        ORDER BY timestamp DESC""",
        (session["user"],)
    )

    predictions = c.fetchall()

    c.execute(
        """SELECT predicted_disease,
        COUNT(*) as count,
        AVG(confidence) as avg_confidence
        FROM prediction_history
        WHERE user_email = ?
        GROUP BY predicted_disease""",
        (session["user"],)
    )

    stats = c.fetchall()

    c.execute(
        """SELECT quiz_type, score, result, timestamp
        FROM quiz_history
        WHERE user_email = ?
        ORDER BY timestamp DESC""",
        (session["user"],)
    )

    quiz_history = c.fetchall()

    c.execute(
        """SELECT quiz_type,
        COUNT(*) as count,
        AVG(score) as avg_score
        FROM quiz_history
        WHERE user_email = ?
        GROUP BY quiz_type""",
        (session["user"],)
    )

    quiz_stats = c.fetchall()

    conn.close()

    return render_template(
        "profile.html",
        predictions=predictions,
        stats=stats,
        quiz_history=quiz_history,
        quiz_stats=quiz_stats,
        user_email=session["user"]
    )

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(
        debug=False,
        host="0.0.0.0",
        port=port
    )
