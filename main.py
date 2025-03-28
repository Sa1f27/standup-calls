import os
import shutil
import hashlib
from datetime import datetime, timedelta
from typing import List, Optional

import jwt
import pyodbc
import whisper

# For Groq usage
# pip install groq
from groq import Groq

from fastapi import (
    FastAPI,
    Request,
    Form,
    UploadFile,
    File,
    HTTPException,
    Depends,
    Query
)
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

#####################
# Helper Functions
#####################

def fetchone_dict(cursor) -> Optional[dict]:
    """
    Fetch a single row as a dictionary.
    Returns None if no row is found.
    """
    row = cursor.fetchone()
    if not row:
        return None
    columns = [desc[0] for desc in cursor.description]
    return dict(zip(columns, row))


def fetchall_dict(cursor) -> List[dict]:
    """
    Fetch all rows as a list of dictionaries.
    Returns an empty list if no rows.
    """
    rows = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]
    result = []
    for row in rows:
        result.append(dict(zip(columns, row)))
    return result

# ==================================================
#  FastAPI & Templating Setup
# ==================================================
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# ==================================================
#  JWT Config
# ==================================================
JWT_SECRET = os.environ.get("JWT_SECRET", "12345668")
JWT_ALGORITHM = "HS256"
JWT_EXP_DELTA_SECONDS = 3600

security = HTTPBearer()

def create_token(email: str, role: str) -> str:
    payload = {
        "email": email,
        "role": role,
        "exp": datetime.utcnow() + timedelta(seconds=JWT_EXP_DELTA_SECONDS),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

def verify_token(token: str) -> dict:
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    return verify_token(credentials.credentials)

# ==================================================
#  SQL Server Connection & DB Setup
# ==================================================
DB_CONFIG = {
    'server': '192.168.48.112',
    'database': 'mentoring_system',
    'username': 'sa',
    'password': 'Welcome@123',
    'driver': '{ODBC Driver 17 for SQL Server}'
}

def get_db_connection():
    conn_str = (
        f"DRIVER={DB_CONFIG['driver']};"
        f"SERVER={DB_CONFIG['server']};"
        f"DATABASE={DB_CONFIG['database']};"
        f"UID={DB_CONFIG['username']};"
        f"PWD={DB_CONFIG['password']}"
    )
    return pyodbc.connect(conn_str)

def setup_database():
    conn = get_db_connection()
    cursor = conn.cursor()
    # Create tables if not exist
    cursor.execute("""
        IF NOT EXISTS (SELECT * FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = 'users')
        BEGIN
            CREATE TABLE users (
                id INT IDENTITY(1,1) PRIMARY KEY,
                email VARCHAR(255) UNIQUE,
                password VARCHAR(255),
                role VARCHAR(50) CHECK (role IN ('mentor', 'student'))
            );
        END
    """)
    cursor.execute("""
        IF NOT EXISTS (SELECT * FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = 'transcripts')
        BEGIN
            CREATE TABLE transcripts (
                id INT IDENTITY(1,1) PRIMARY KEY,
                mentor_email VARCHAR(255),
                batch VARCHAR(255),
                transcript TEXT,
                upload_date DATETIME DEFAULT GETDATE()
            );
        END
    """)
    cursor.execute("""
        IF NOT EXISTS (SELECT * FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = 'results')
        BEGIN
            CREATE TABLE results (
                id INT IDENTITY(1,1) PRIMARY KEY,
                student_email VARCHAR(255),
                batch VARCHAR(255),
                results TEXT,
                ai_feedback TEXT,
                submission_date DATETIME DEFAULT GETDATE()
            );
        END
    """)
    conn.commit()
    cursor.close()
    conn.close()

# Initialize DB
setup_database()

# ==================================================
#  Utility & Groq Integration
# ==================================================
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

try:
    groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))
except Exception:
    groq_client = None

def generate_questions_with_groq(transcript: str) -> List[str]:
    if not groq_client:
        return [
            "1) What is the main topic discussed in this lecture?",
            "2) List two key points mentioned by the speaker.",
            "3) How does this topic relate to your overall course?",
            "4) Mention any real-world example provided (or give one).",
            "5) Summarize the lecture in your own words."
        ]

    prompt = f"""
You are an expert educator. Based on the following transcript, generate 5 quiz questions 
to test a student's understanding. The questions should be numbered 1 to 5, with no extra commentary.

Transcript:
\"\"\"{transcript}\"\"\"
"""
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "Generate 5 quiz questions from the transcript."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=512,
            top_p=1
        )
        raw = response.choices[0].message.content.strip()
        lines = [x.strip() for x in raw.split("\n") if x.strip()]
        return lines[:5]
    except Exception:
        return [
            "1) What is the main topic discussed in this lecture?",
            "2) List two key points mentioned by the speaker.",
            "3) How does this topic relate to your overall course?",
            "4) Mention any real-world example provided (or give one).",
            "5) Summarize the lecture in your own words."
        ]

def evaluate_answers_with_groq(answers_text: str, transcript: str) -> str:
    if not groq_client:
        return "Good attempt! (Groq not configured; default feedback.)"

    prompt = f"""
You are an expert educator. Evaluate the following student answers in relation to the transcript.
Provide a concise summary of correctness, clarity, and missing pieces. Then give an overall score.

Transcript:
\"\"\"{transcript}\"\"\"

Student Answers:
\"\"\"{answers_text}\"\"\"
"""
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "Evaluate student quiz answers."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=512,
            top_p=1
        )
        feedback = response.choices[0].message.content.strip()
        return feedback
    except Exception:
        return "Could not evaluate with Groq. Default feedback: Good attempt!"

# ==================================================
#  Speech-to-Text (STT) Endpoint (Server-Side)
# ==================================================
@app.post("/stt_whisper")
async def stt_whisper_endpoint(file: UploadFile = File(...)):
    """
    (Optional) Endpoint that does server-side STT using Whisper.
    """
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    text = ""
    try:
        model = whisper.load_model("tiny")
        result = model.transcribe(temp_path)
        text = result.get("text", "")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription error: {e}")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    return {"transcript": text}

# ==================================================
#  Auth & Index
# ==================================================
@app.get("/", response_class=HTMLResponse)
async def get_home(request: Request):
    """
    Landing page, shows login/register (index.html).
    """
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/login")
async def post_login(
    email: str = Form(...),
    password: str = Form(...),
    role: str = Form(...)
):
    """
    Login endpoint that checks user/password/role and redirects to the
    appropriate dashboard with a JWT token.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    hashed_pw = hash_password(password)

    try:
        query = "SELECT * FROM users WHERE email = ? AND password = ?"
        cursor.execute(query, (email, hashed_pw))
        user = fetchone_dict(cursor)
    except pyodbc.Error as e:
        cursor.close()
        conn.close()
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    cursor.close()
    conn.close()

    if not user or user["role"] != role:
        raise HTTPException(status_code=400, detail="Invalid credentials or role")

    token = create_token(email, role)
    if role == "mentor":
        return RedirectResponse(url=f"/mentor/dashboard?token={token}", status_code=303)
    return RedirectResponse(url=f"/student/dashboard?token={token}", status_code=303)

@app.post("/register")
async def post_register(
    email: str = Form(...),
    password: str = Form(...),
    role: str = Form(...)
):
    """
    Registration endpoint that inserts a new user into the DB.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    hashed_pw = hash_password(password)

    try:
        insert_query = "INSERT INTO users (email, password, role) VALUES (?, ?, ?)"
        cursor.execute(insert_query, (email, hashed_pw, role))
        conn.commit()
    except pyodbc.Error as e:
        cursor.close()
        conn.close()
        # Could parse for duplicate error, but let's just raise generic.
        raise HTTPException(status_code=400, detail="User already exists or DB error")

    cursor.close()
    conn.close()
    return RedirectResponse(url="/", status_code=303)

# ==================================================
#  Mentor Routes
# ==================================================
@app.get("/mentor/dashboard", response_class=HTMLResponse)
async def mentor_dashboard(request: Request, token: str):
    """
    Mentor dashboard page showing transcript upload form, etc.
    """
    user = verify_token(token)
    if user["role"] != "mentor":
        raise HTTPException(status_code=403, detail="Access denied")

    return templates.TemplateResponse("mentor_dashboard.html", {
        "request": request,
        "token": token,
        "user": user
    })

@app.post("/mentor/upload")
async def mentor_upload(
    request: Request,
    token: str = Form(...),
    batch: str = Form(...),
    video: UploadFile = File(...)
):
    """
    Mentor transcript upload route that uses Whisper to transcribe
    the uploaded video and store transcript in DB.
    """
    user = verify_token(token)
    if user["role"] != "mentor":
        raise HTTPException(status_code=403, detail="Not authorized")

    temp_path = f"temp_{video.filename}"
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(video.file, f)

    try:
        model = whisper.load_model("tiny")
        result = model.transcribe(temp_path)
        transcript = result.get("text", "")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        insert_query = "INSERT INTO transcripts (mentor_email, batch, transcript) VALUES (?, ?, ?)"
        cursor.execute(insert_query, (user["email"], batch, transcript))
        conn.commit()
    except pyodbc.Error as e:
        cursor.close()
        conn.close()
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    cursor.close()
    conn.close()

    return RedirectResponse(url=f"/mentor/dashboard?token={token}", status_code=303)

@app.get("/mentor/results", response_class=HTMLResponse)
async def mentor_view_results(
    request: Request,
    token: str,
    batch: Optional[str] = Query(None)
):
    """
    Mentor can view quiz results for a specific batch.
    """
    user = verify_token(token)
    if user["role"] != "mentor":
        raise HTTPException(status_code=403, detail="Not authorized")

    results_data = []
    if batch:
        conn = get_db_connection()
        cursor = conn.cursor()
        try:
            query = "SELECT student_email, results, ai_feedback, submission_date FROM results WHERE batch = ?"
            cursor.execute(query, (batch,))
            results_data = fetchall_dict(cursor)
        except pyodbc.Error as e:
            cursor.close()
            conn.close()
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
        cursor.close()
        conn.close()

    return templates.TemplateResponse("mentor_results.html", {
        "request": request,
        "token": token,
        "user": user,
        "batch": batch,
        "results_data": results_data
    })

# ==================================================
#  Student Routes
# ==================================================
@app.get("/student/dashboard", response_class=HTMLResponse)
async def student_dashboard(request: Request, token: str):
    """
    Student dashboard, lists available quiz batches, etc.
    """
    user = verify_token(token)
    if user["role"] != "student":
        raise HTTPException(status_code=403, detail="Access denied")

    return templates.TemplateResponse("student_dashboard.html", {
        "request": request,
        "token": token,
        "user": user
    })

@app.get("/student/quiz", response_class=HTMLResponse)
async def get_quiz(
    request: Request,
    token: str,
    batch: str
):
    """
    Student gets the quiz page for a particular batch
    (fetches latest transcript, generates questions).
    """
    user = verify_token(token)
    if user["role"] != "student":
        raise HTTPException(status_code=403, detail="Access denied")

    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        query = "SELECT TOP 1 transcript FROM transcripts WHERE batch = ? ORDER BY upload_date DESC"
        cursor.execute(query, (batch,))
        record = fetchone_dict(cursor)
    except pyodbc.Error as e:
        cursor.close()
        conn.close()
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    cursor.close()
    conn.close()

    if not record:
        raise HTTPException(status_code=404, detail=f"No transcript found for batch '{batch}'")

    transcript_text = record["transcript"]
    questions = generate_questions_with_groq(transcript_text)

    return templates.TemplateResponse("quiz.html", {
        "request": request,
        "token": token,
        "user": user,
        "batch": batch,
        "transcript": transcript_text,
        "questions": questions,
        "submitted": False,
        "results": None
    })

from typing import List

@app.post("/student/quiz", response_class=HTMLResponse)
async def post_quiz(
    request: Request,
    token: str = Form(...),
    batch: str = Form(...),
    answers: List[str] = Form(...)
):
    """
    Student submits answers for a quiz, we then evaluate them (Groq or fallback),
    store them in DB, and display the feedback on the same page.
    """
    user = verify_token(token)
    if user["role"] != "student":
        raise HTTPException(status_code=403, detail="Access denied")

    # Re-fetch the transcript
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        query = "SELECT TOP 1 transcript FROM transcripts WHERE batch = ? ORDER BY upload_date DESC"
        cursor.execute(query, (batch,))
        record = fetchone_dict(cursor)
    except pyodbc.Error as e:
        cursor.close()
        conn.close()
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    transcript_text = record["transcript"] if record else ""

    # Combine answers for storage
    combined_answers = ""
    for i, ans in enumerate(answers, start=1):
        combined_answers += f"Q{i}: {ans}\n"

    # Evaluate answers with Groq
    ai_feedback = evaluate_answers_with_groq(combined_answers, transcript_text)

    # Save results
    try:
        insert_query = "INSERT INTO results (student_email, batch, results, ai_feedback) VALUES (?, ?, ?, ?)"
        cursor.execute(insert_query, (user["email"], batch, combined_answers, ai_feedback))
        conn.commit()
    except pyodbc.Error as e:
        cursor.close()
        conn.close()
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    cursor.close()
    conn.close()

    # Display same quiz page, but with results
    questions = generate_questions_with_groq(transcript_text)

    return templates.TemplateResponse("quiz.html", {
        "request": request,
        "token": token,
        "user": user,
        "batch": batch,
        "transcript": transcript_text,
        "questions": questions,
        "submitted": True,
        "results": {
            "answers": combined_answers,
            "ai_feedback": ai_feedback
        }
    })
