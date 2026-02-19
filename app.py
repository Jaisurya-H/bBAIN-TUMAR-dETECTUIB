from flask import Flask, render_template, request, redirect, session, send_file
import sqlite3
import os
import cv2
import numpy as np
import pickle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

app = Flask(__name__)
app.secret_key = "healtech_secure_key"

# Database Initialization
def init_db():
    conn = sqlite3.connect("database.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            email TEXT UNIQUE,
            password TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            result TEXT,
            confidence REAL,
            model_used TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

# Load Models
try:
    lr_model = pickle.load(open("model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
except FileNotFoundError:
    print("Warning: model.pkl or scaler.pkl not found.")
    lr_model = None
    scaler = None

if os.path.exists("svm_model.pkl"):
    svm_model = pickle.load(open("svm_model.pkl", "rb"))
else:
    svm_model = None

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Helper Function for Risk Level
def assess_risk(prediction, confidence):
    if "No Tumor" in prediction:
        return "risk-low", "Low Risk", "The scan shows no signs of tumor. Regular check-ups are advised."
    elif confidence > 90:
        return "risk-high", "High Risk", f"The scan indicates a high probability of {prediction}. Immediate clinical correlation is recommended."
    else:
        return "risk-moderate", "Moderate Risk", f"The scan suggests possible {prediction} characteristics. Further testing is required."

# --- Routes ---

@app.route("/")
def home():
    if "user" in session:
        return redirect("/dashboard")
    return redirect("/login")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        print(f"DEBUG: Attempting login for email: {email}") # DEBUG

        conn = sqlite3.connect("database.db")
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE email=? AND password=?", (email, password))
        user = c.fetchone()
        conn.close()

        if user:
            print(f"DEBUG: Login successful for user: {user[1]}") # DEBUG
            session["user"] = user[0]
            session["name"] = user[1]
            return redirect("/dashboard")
        else:
            print("DEBUG: Login failed - Invalid credentials") # DEBUG
            return render_template("login.html", error="Invalid credentials") # Pass error to template

    return render_template("login.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form["name"]
        email = request.form["email"]
        password = request.form["password"]

        conn = sqlite3.connect("database.db")
        c = conn.cursor()
        try:
            c.execute("INSERT INTO users (name,email,password) VALUES (?,?,?)", (name, email, password))
            conn.commit()
        except sqlite3.IntegrityError:
            return "Email already exists!" # Basic error handling
        finally:
            conn.close()

        return redirect("/login")

    return render_template("register.html")

@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect("/login")

    conn = sqlite3.connect("database.db")
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM reports WHERE user_id=?", (session["user"],))
    total = c.fetchone()[0]
    conn.close()

    return render_template("dashboard.html", name=session["name"], total=total)

@app.route("/analyze", methods=["GET", "POST"])
def analyze():
    if "user" not in session:
        return redirect("/login")
        
    if request.method == "GET":
        return render_template("analyze.html")

    if not lr_model or not scaler:
         return render_template("analyze.html", prediction="Error: Model not loaded")

    file = request.files["file"]
    model_choice = request.form.get("model_choice", "lr")
    
    if file.filename == "":
        return render_template("analyze.html", prediction="No file selected")

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    # --- Preprocessing ---
    img = cv2.imread(filepath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (200, 200))
    img = cv2.medianBlur(img, 5)
    img = img.flatten().reshape(1, -1)
    img = scaler.transform(img)

    # --- Model Selection ---
    if model_choice == "svm" and svm_model:
        model = svm_model
        model_name = "Support Vector Machine"
    else:
        model = lr_model
        model_name = "Logistic Regression"

    prediction = model.predict(img)[0]
    probabilities = model.predict_proba(img)[0]
    confidence = round(np.max(probabilities) * 100, 2)

    labels = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
    result = labels[prediction]
    
    # Risk Assessment
    risk_class, risk_label, summary_text = assess_risk(result, confidence)

    # Save to DB
    conn = sqlite3.connect("database.db")
    c = conn.cursor()
    c.execute("INSERT INTO reports (user_id, result, confidence, model_used) VALUES (?,?,?,?)",
              (session["user"], result, confidence, model_name))
    conn.commit()
    conn.close()

    return render_template("analyze.html",
                           prediction=result,
                           confidence=confidence,
                           model_used=model_name,
                           image_path=filepath,
                           risk_class=risk_class,
                           risk_label=risk_label,
                           summary_text=summary_text)

@app.route("/history")
def history():
    if "user" not in session:
        return redirect("/login")
        
    conn = sqlite3.connect("database.db")
    c = conn.cursor()
    c.execute("SELECT result, confidence, model_used FROM reports WHERE user_id=?", (session["user"],))
    rows = c.fetchall()
    
    # Convert tuples to dict functionality for template
    history_data = [{"result": r[0], "confidence": r[1], "model": r[2]} for r in rows]
    conn.close()

    return render_template("history.html", history=history_data) # Reusing history.html, might need slight tweak if it expects sidebar

@app.route("/logout")
def logout():
    session.clear()
    return redirect("/login")


@app.route("/download_report/<result>/<confidence>")
def download_report(result, confidence):
    if "user" not in session:
        return redirect("/login")
        
    filename = "report.pdf"
    doc = SimpleDocTemplate(filename)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("HealTech Brain Tumor Report", styles["Title"]))
    elements.append(Paragraph(f"Patient ID: {session['user']}", styles["Normal"]))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(f"Prediction: {result}", styles["Normal"]))
    elements.append(Paragraph(f"Confidence: {confidence}%", styles["Normal"]))
    
    risk_class, risk_label, summary_text = assess_risk(result, float(confidence))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(f"Assessment: {risk_label}", styles["Heading3"]))
    elements.append(Paragraph(summary_text, styles["Normal"]))

    elements.append(Spacer(1, 12))
    elements.append(Paragraph("Disclaimer: This is an AI-generated report.", styles["Italic"]))

    doc.build(elements)

    return send_file(filename, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
