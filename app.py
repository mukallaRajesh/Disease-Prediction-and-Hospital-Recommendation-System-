import os
from pathlib import Path
from flask import Flask, request, jsonify, render_template, redirect, url_for, session, flash
import pandas as pd
from collections import Counter
import joblib
import logging
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

BASE_DIR = Path(__file__).resolve().parent

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "change-this-secret-key")
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

    dis_sym_data_v1 = pd.read_csv(
        BASE_DIR / "Processed_Dataset1.csv"
    )

    doc_data = pd.read_csv(
        BASE_DIR / "Doctor_Versus_Disease.csv",
        encoding="latin1",
        names=["Disease", "Specialist"]
    )

    des_data = pd.read_csv(
        BASE_DIR / "Disease_Description.csv"
    )

    algorithms = joblib.load(
        BASE_DIR / "trained_algorithms.pkl"
    )

    le = joblib.load(
        BASE_DIR / "label_encoder.pkl"
    )

    chatbot_file = BASE_DIR / "chatbot.csv"

    if not chatbot_file.exists():
        chatbot_file = BASE_DIR / "chatbot final ques dataset.csv"

    if not chatbot_file.exists():
        raise FileNotFoundError(
            "Chatbot dataset not found. Add chatbot.csv or chatbot final ques dataset.csv."
        )

    chatbot_df = pd.read_csv(
        chatbot_file,
        encoding="latin1"
    )

    if "User Question" not in chatbot_df.columns:
        raise ValueError(
            "Chatbot dataset must contain a 'User Question' column."
        )

    if "AI Response" not in chatbot_df.columns:
        raise ValueError(
            "Chatbot dataset must contain an 'AI Response' column."
        )

    chatbot_df["User Question"] = (
        chatbot_df["User Question"]
        .fillna("")
        .astype(str)
    )

    chatbot_df["AI Response"] = (
        chatbot_df["AI Response"]
        .fillna("")
        .astype(str)
    )

    chatbot_df = chatbot_df[
        chatbot_df["User Question"].str.strip() != ""
    ].reset_index(drop=True)

    chatbot_vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        max_features=20000,
        ngram_range=(1, 2)
    )

    chatbot_matrix = chatbot_vectorizer.fit_transform(
        chatbot_df["User Question"]
    )

    chatbot_df["Answer"] = chatbot_df["AI Response"]

    dis_sym_data_v1 = dis_sym_data_v1.loc[
        :,
        ~dis_sym_data_v1.columns.str.contains("^Unnamed")
    ]

    test_col = [
        col for col in dis_sym_data_v1.columns
        if col != "Disease"
    ]

    logging.info("Datasets and models loaded successfully!")

except Exception as e:
    logging.error(
        f"Error loading models or datasets: {e}"
    )
    raise

def login_required(f):
    def decorated_function(*args, **kwargs):
        if "user" not in session:
            flash(
                "Please login to access this page.",
                "error"
            )
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
        flash(
            "Please fill in all fields.",
            "error"
        )
        return redirect(url_for("signup"))

    if password != confirm_password:
        flash(
            "Passwords do not match.",
            "error"
        )
        return redirect(url_for("signup"))

    conn = sqlite3.connect(
        BASE_DIR / "users.db"
    )

    c = conn.cursor()

    c.execute(
        "SELECT email FROM users WHERE email = ?",
        (email,)
    )

    if c.fetchone():
        flash(
            "Email already registered.",
            "error"
        )
        conn.close()
        return redirect(url_for("signup"))

    hashed_password = generate_password_hash(
        password
    )

    c.execute(
        "INSERT INTO users (email, password) VALUES (?, ?)",
        (email, hashed_password)
    )

    conn.commit()
    conn.close()

    flash(
        "Registration successful! Please login.",
        "success"
    )

    return redirect(
        url_for("login_page")
    )

@app.route("/login")
def login_page():
    if "user" in session:
        return redirect(
            url_for("predict_page")
        )

    return render_template("login.html")

@app.route("/login", methods=["POST"])
def login():
    email = request.form.get("email")
    password = request.form.get("password")

    if not email or not password:
        flash(
            "Please fill in all fields.",
            "error"
        )
        return redirect(
            url_for("login_page")
        )

    conn = sqlite3.connect(
        BASE_DIR / "users.db"
    )

    c = conn.cursor()

    c.execute(
        "SELECT password FROM users WHERE email = ?",
        (email,)
    )

    result = c.fetchone()

    conn.close()

    if result and check_password_hash(
        result[0],
        password
    ):
        session["user"] = email

        flash(
            "Login successful!",
            "success"
        )

        return redirect(
            url_for("predict_page")
        )

    flash(
        "Invalid email or password.",
        "error"
    )

    return redirect(
        url_for("login_page")
    )

@app.route("/logout")
def logout():
    session.pop(
        "user",
        None
    )

    session.pop(
        "chat_history",
        None
    )

    flash(
        "You have been logged out.",
        "success"
    )

    return redirect(
        url_for("home")
    )

@app.route("/predict")
@login_required
def predict_page():
    symptoms = test_col

    return render_template(
        "prediction.html",
        symptoms=symptoms
    )

@app.route("/predict", methods=["POST"])
@login_required
def predict():
    try:
        data = request.get_json(
            silent=True
        ) or {}

        symptoms = data.get(
            "symptoms",
            []
        )

        if not symptoms:
            return jsonify(
                {
                    "error": "No symptoms provided."
                }
            ), 400

        formatted_symptoms = []

        for symptom in symptoms:
            formatted_symptom = (
                str(symptom)
                .lower()
                .replace(" ", "_")
            )

            formatted_symptom = "".join(
                c
                for c in formatted_symptom
                if c.isalnum() or c == "_"
            )

            formatted_symptoms.append(
                formatted_symptom
            )

        test_data = {
            col: 1 if col in formatted_symptoms else 0
            for col in test_col
        }

        test_df = pd.DataFrame(
            [test_data]
        )

        predicted = []

        for model_name, values in algorithms.items():
            model = values["model"]

            predict_disease = model.predict(
                test_df
            )

            predict_disease = le.inverse_transform(
                predict_disease
            )

            predicted.extend(
                predict_disease
            )

        disease_counts = Counter(
            predicted
        )

        percentage_per_disease = {
            disease: (
                count / len(algorithms)
            ) * 100
            for disease, count
            in disease_counts.items()
        }

        result_df = pd.DataFrame(
            {
                "Disease": list(
                    percentage_per_disease.keys()
                ),
                "Chances": list(
                    percentage_per_disease.values()
                )
            }
        )

        result_df = result_df.sort_values(
            "Chances",
            ascending=False
        ).reset_index(
            drop=True
        )

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

        if not result_df.empty:
            top_prediction = result_df.iloc[0]

            conn = sqlite3.connect(
                BASE_DIR / "users.db"
            )

            c = conn.cursor()

            c.execute(
                """INSERT INTO prediction_history
                (user_email, symptoms, predicted_disease, confidence)
                VALUES (?, ?, ?, ?)""",
                (
                    session["user"],
                    ",".join(
                        map(str, symptoms)
                    ),
                    str(
                        top_prediction["Disease"]
                    ),
                    float(
                        top_prediction["Chances"]
                    )
                )
            )

            conn.commit()
            conn.close()

        results = []

        for _, row in result_df.iterrows():
            results.append(
                {
                    "Disease": row["Disease"],
                    "Chances": float(
                        row["Chances"]
                    ),
                    "Specialist": row.get(
                        "Specialist",
                        "Specialist info not available"
                    ),
                    "Description": row.get(
                        "Description",
                        "Description not available"
                    )
                }
            )

        return jsonify(results)

    except Exception as e:
        logging.error(
            f"Error in prediction: {e}"
        )

        return jsonify(
            {
                "error": str(e)
            }
        ), 500

@app.route("/quiz/heart")
@login_required
def heart_quiz():
    return render_template(
        "quiz_heart.html"
    )

@app.route("/quiz/lung")
@login_required
def lung_quiz():
    return render_template(
        "quiz_lung.html"
    )

@app.route("/quiz/heart", methods=["POST"])
@login_required
def heart_quiz_submit():
    try:
        data = request.get_json(
            silent=True
        ) or {}

        score = data.get(
            "score",
            0
        )

        result = data.get(
            "result",
            ""
        )

        conn = sqlite3.connect(
            BASE_DIR / "users.db"
        )

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

        return jsonify(
            {
                "success": True,
                "message": "Quiz result saved successfully"
            }
        )

    except Exception as e:
        logging.error(
            f"Error saving heart quiz result: {e}"
        )

        return jsonify(
            {
                "error": str(e)
            }
        ), 500

@app.route("/quiz/lung", methods=["POST"])
@login_required
def lung_quiz_submit():
    try:
        data = request.get_json(
            silent=True
        ) or {}

        score = data.get(
            "score",
            0
        )

        result = data.get(
            "result",
            ""
        )

        conn = sqlite3.connect(
            BASE_DIR / "users.db"
        )

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

        return jsonify(
            {
                "success": True,
                "message": "Quiz result saved successfully"
            }
        )

    except Exception as e:
        logging.error(
            f"Error saving lung quiz result: {e}"
        )

        return jsonify(
            {
                "error": str(e)
            }
        ), 500

@app.route("/chatbot")
@login_required
def chatbot():
    if "chat_history" not in session:
        session["chat_history"] = []

    return render_template(
        "chatbot.html",
        chat_history=session["chat_history"]
    )

def find_most_similar_answer(
    user_question
):
    user_vector = chatbot_vectorizer.transform(
        [str(user_question)]
    )

    similarities = cosine_similarity(
        user_vector,
        chatbot_matrix
    ).flatten()

    most_similar_idx = similarities.argmax()

    most_similar_row = chatbot_df.iloc[
        most_similar_idx
    ]

    return (
        most_similar_row["User Question"],
        most_similar_row["Answer"]
    )

def generate_openai_response(
    chat_history,
    relevant_question,
    relevant_answer
):
    if openai_client is None:
        raise RuntimeError(
            "OPENAI_API_KEY is not configured in Render Environment Variables."
        )

    prompt = (
        "You are a professional healthcare chatbot. "
        "Provide general health information only. "
        "Do not claim to diagnose the user. "
        "For serious or emergency symptoms, advise the user "
        "to seek appropriate medical care. "
        "Answer in 2-3 concise points.\n\n"
    )

    recent_history = chat_history[-6:]

    for entry in recent_history:
        prompt += (
            f"User: {entry['user']}\n"
            f"Bot: {entry['bot']}\n"
        )

    prompt += (
        f"\nRelevant question from knowledge base: "
        f"{relevant_question}\n\n"
        f"Relevant information: {relevant_answer}\n\n"
        "Answer the user's latest question directly. "
        "Use simple professional language. "
        "Do not mention the internal knowledge base.\n\n"
        "Bot:"
    )

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
        data = request.get_json(
            silent=True
        ) or {}

        question = data.get(
            "question",
            ""
        )

        question = str(
            question
        ).strip()

        if not question:
            return jsonify(
                {
                    "error": "No question provided"
                }
            ), 400

        if "chat_history" not in session:
            session["chat_history"] = []

        relevant_question, relevant_answer = (
            find_most_similar_answer(
                question
            )
        )

        session["chat_history"].append(
            {
                "user": question,
                "bot": relevant_answer
            }
        )

        enhanced_answer = generate_openai_response(
            session["chat_history"],
            relevant_question,
            relevant_answer
        )

        session["chat_history"][-1]["bot"] = (
            enhanced_answer
        )

        session["chat_history"] = (
            session["chat_history"][-10:]
        )

        session.modified = True

        return jsonify(
            {
                "response": enhanced_answer
            }
        )

    except Exception as e:
        logging.error(
            f"Error in chatbot: {e}"
        )

        return jsonify(
            {
                "error": str(e)
            }
        ), 500

@app.route("/profile")
@login_required
def profile():
    conn = sqlite3.connect(
        BASE_DIR / "users.db"
    )

    c = conn.cursor()

    c.execute(
        """SELECT predicted_disease,
        confidence,
        timestamp,
        symptoms
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
        """SELECT quiz_type,
        score,
        result,
        timestamp
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
    port = int(
        os.getenv(
            "PORT",
            5000
        )
    )

    app.run(
        debug=False,
        host="0.0.0.0",
        port=port
    )