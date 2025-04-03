import os
import shutil
import hashlib
import tempfile
import gridfs
from datetime import datetime, timedelta
from typing import List, Optional

import jwt
import pyodbc
import whisper
from pymongo import MongoClient

# Set your OpenAI API key
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

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
    IF NOT EXISTS (SELECT * FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = 'daily_questions')
        BEGIN
            CREATE TABLE daily_questions (
                id INT IDENTITY(1,1) PRIMARY KEY,
                batch VARCHAR(255),
                question TEXT,
                transcript_hash VARCHAR(255),
                created_at DATETIME DEFAULT GETDATE()
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
                name VARCHAR(255),
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

# ===================================================
# MongoDB connection
# ===================================================
def get_mongo_connection():
    """Establish connection to MongoDB"""
    print("DEBUG: Starting connection to MongoDB...")
    client = MongoClient('mongodb://192.168.48.112:27017/')
    db = client.video_db
    print("DEBUG: MongoDB connection established.")
    return db, gridfs.GridFS(db)

def fetch_video_by_batch_and_date(batch, date=None):
    """
    Fetch video from MongoDB by batch and date.
    If date is None, fetch the latest video for the batch.
    
    Args:
        batch (str): Batch name (e.g., 'batch_03')
        date (str, optional): Date in format 'YYYY-MM-DD'. Defaults to None.
    
    Returns:
        tuple: (video_binary, file_info, error_message)
    """
    print(f"DEBUG: fetch_video_by_batch_and_date called with batch='{batch}' and date='{date}'")
    db, fs = get_mongo_connection()
    
    # Determine the collection name
    batch_collection = batch
    print(f"DEBUG: Using collection '{batch_collection}' for batch '{batch}'")
    
    # Check if collection exists
    if batch_collection in db.list_collection_names():
        print(f"DEBUG: Collection '{batch_collection}' found.")
        collection = db[batch_collection]
    else:
        error_msg = f"Collection {batch_collection} not found in MongoDB."
        print(f"ERROR: {error_msg}")
        return None, None, error_msg
    
    # Build query based on date parameter
    query = {}
    if date:
        print(f"DEBUG: Processing date parameter: {date}")
        try:
            # Parse the date and create a date range for that day
            date_obj = datetime.strptime(date, '%Y-%m-%d')
            date_start = date_obj.replace(hour=0, minute=0, second=0)
            date_end = date_obj.replace(hour=23, minute=59, second=59)
            print(f"DEBUG: Parsed date_start: {date_start}, date_end: {date_end}")
            
            # Build query conditions for both datetime and string-based fields
            query_conditions = [
                {"uploadDate": {"$gte": date_start, "$lte": date_end}},
                {"upload_datetime": {"$regex": f"^{date}"}}
            ]
            query = {"$or": query_conditions}
            print(f"DEBUG: Built query: {query}")
        except ValueError as ve:
            error_msg = "Invalid date format. Please use YYYY-MM-DD."
            print(f"ERROR: {error_msg} Exception: {ve}")
            return None, None, error_msg
    
    # Find the file document using the query
    sort_criteria = [("uploadDate", -1)]  # Sort by upload date descending
    print("DEBUG: Querying the collection for the file document...")
    file_doc = collection.find_one(query, sort=sort_criteria)
    
    if not file_doc:
        error_msg = f"No video found for batch {batch}" + (f" on date {date}" if date else "")
        print(f"ERROR: {error_msg}")
        return None, None, error_msg
    
    print(f"DEBUG: Found file document: {file_doc}")
    
    try:
        # If this is a GridFS file, get the file from GridFS
        if "_id" in file_doc and fs.exists(file_doc["_id"]):
            print("DEBUG: File found in GridFS by _id. Retrieving file...")
            grid_out = fs.get(file_doc["_id"])
            file_data = grid_out.read()
            print("DEBUG: File retrieved successfully via _id.")
            return file_data, file_doc, None
        elif "video_id" in file_doc and fs.exists(file_doc["video_id"]):
            print("DEBUG: File found in GridFS by video_id. Retrieving file...")
            grid_out = fs.get(file_doc["video_id"])
            file_data = grid_out.read()
            print("DEBUG: File retrieved successfully via video_id.")
            return file_data, file_doc, None
        else:
            # Check if we're using a chunks collection for large files
            print("DEBUG: Attempting to reconstruct file from chunks...")
            chunks_collection = db[f"{batch}.chunks"]
            if chunks_collection.count_documents({"files_id": file_doc["_id"]}) > 0:
                chunks = chunks_collection.find({"files_id": file_doc["_id"]}).sort("n", 1)
                file_data = b''.join(chunk["data"] for chunk in chunks)
                print("DEBUG: File reconstructed from chunks successfully.")
                return file_data, file_doc, None
            else:
                print("DEBUG: No chunks found for file reconstruction.")
    except Exception as e:
        error_msg = f"Error retrieving file: {str(e)}"
        print(f"ERROR: {error_msg}")
        return None, None, error_msg
    
    error_msg = "File exists in database but couldn't be retrieved."
    print(f"ERROR: {error_msg}")
    return None, None, error_msg

def transcribe_mongo_video(file_data, file_info):
    """
    Transcribe video from MongoDB using Whisper

    Args:
        file_data (bytes): Binary data of the video
        file_info (dict): File metadata

    Returns:
        str: Transcription text
    """
    print("DEBUG: Starting transcription process...")

    # Use file_id or _id for safe filename
    file_identifier = str(file_info.get('file_id', file_info.get('_id', 'temp_video')))
    file_name = file_identifier + '.mp4'
    print(f"DEBUG: Using safe file name: {file_name}")

    # Create a temporary directory and path
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, file_name)
    print(f"DEBUG: Temporary file path created at: {temp_path}")

    try:
        # Write the binary data to the temporary file
        with open(temp_path, 'wb') as f:
            f.write(file_data)
        print("DEBUG: Video data written to temporary file.")

        # Transcribe using Whisper
        print("DEBUG: Loading Whisper model for transcription...")
        model = whisper.load_model("tiny")
        print("DEBUG: Whisper model loaded. Starting transcription...")
        result = model.transcribe(temp_path)
        transcript = result.get("text", "")
        print("DEBUG: Transcription complete.")
        return transcript
    except Exception as e:
        print(f"ERROR: Error during transcription: {str(e)}")
        return ""
    finally:
        if os.path.exists(temp_dir):
            print("DEBUG: Cleaning up temporary files...")
            shutil.rmtree(temp_dir)
            print("DEBUG: Temporary files removed.")


# ==================================================
#  Utility & OpenAI Integration
# ==================================================
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

# Generate quiz questions using OpenAI
def generate_questions_with_openai(transcript: str) -> List[str]:
    prompt = f"""
You are a highly skilled teacher. Given the transcript below, create 10 thoughtful quiz questions that test a student's understanding.

Guidelines:
- Ask about key concepts, facts, and implications.
- Questions should be clear and open-ended or short-answer style.
- Do NOT include answers, numbering, or any commentary—just the raw questions.

Transcript:
\"\"\"{transcript}\"\"\"
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You generate quiz questions for students based on transcripts."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=512
    )
    raw = response.choices[0].message.content.strip()
    lines = [x.strip() for x in raw.split("\n") if x.strip()]
    return lines[:10]


# Evaluation using OpenAI
def evaluate_answers_with_openai(answers_text: str, questions: str) -> str:
    prompt = f"""
You are an experienced educator grading student answers.

Instructions:
- Evaluate each student answer in context of the matching question.
- Comment on accuracy, completeness, clarity, and reasoning.
- After evaluating, provide a final score out of 10 and brief overall feedback.

Questions:
\"\"\"{questions}\"\"\"

Student Answers:
\"\"\"{answers_text}\"\"\"
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a teacher grading student quiz submissions."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=512
    )
    return response.choices[0].message.content.strip()


# ==================================================
#  Random Question Generation from docx
# ==================================================
import random
from docx import Document

def ask_random_questions():
    number_of_questions = 10
    docx_file_path = r"clean_questions.docx"
    
    # Load the DOCX file
    doc = Document(docx_file_path)
    
    # Collect all non-empty paragraphs
    questions = []
    for para in doc.paragraphs:
        text = para.text.strip()
        if text:
            questions.append(text)
    
    # Select random questions
    if len(questions) >= number_of_questions:
        random_questions = random.sample(questions, number_of_questions)
        
        # Format questions as numbered strings in a list
        formatted_questions = [f"{i}. {question}" for i, question in enumerate(random_questions, 1)]
        
        return formatted_questions
    else:
        print("Not enough questions found")
        return []

@app.get("/api/placeholder/{width}/{height}")
async def get_placeholder(width: int, height: int):
    return {"message": f"No placeholder at {width}x{height}"}

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
        model = whisper.load_model("small") # tiny, base, small, medium, large
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

@app.post("/mentor/fetch_transcript")
async def fetch_transcript(
    request: Request,
    token: str = Form(...),
    batch: str = Form(...),
    upload_date: Optional[str] = Form(None)
):
    """
    Mentor fetch transcript route that retrieves a video from MongoDB,
    transcribes it using Whisper, and stores the transcript in the SQL DB.
    """
    # Verify user token and role
    user = verify_token(token)
    if user["role"] != "mentor":
        raise HTTPException(status_code=403, detail="Not authorized")
    
    # Extract batch name (removing .files if present)
    batch_name = batch.split('.')[0] if '.files' in batch else batch
    
    # Fetch video from MongoDB
    file_data, file_info, error_message = fetch_video_by_batch_and_date(batch_name, upload_date)
    
    if error_message:
        # Return to dashboard with error message
        return RedirectResponse(
            url=f"/mentor/dashboard?token={token}&error={error_message}",
            status_code=303
        )
    
    if not file_data or not file_info:
        return RedirectResponse(
            url=f"/mentor/dashboard?token={token}&error=No video file found",
            status_code=303
        )
    
    # Transcribe the video
    transcript = transcribe_mongo_video(file_data, file_info)
    
    if not transcript:
        return RedirectResponse(
            url=f"/mentor/dashboard?token={token}&error=Failed to transcribe video",
            status_code=303
        )
    
    # Convert batch format if needed (batch_03 to b3)
    if batch_name.startswith('batch_'):
        batch_num = int(batch_name.split('_')[1])
        sql_batch = f"b{batch_num}"
    else:
        sql_batch = batch_name
    
    # Save transcript to SQL database
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        insert_query = "INSERT INTO transcripts (mentor_email, batch, transcript) VALUES (?, ?, ?)"
        cursor.execute(insert_query, (user["email"], sql_batch, transcript))
        conn.commit()
    except pyodbc.Error as e:
        cursor.close()
        conn.close()
        return RedirectResponse(
            url=f"/mentor/dashboard?token={token}&error=Database error: {str(e)}",
            status_code=303
        )
    finally:
        cursor.close()
        conn.close()
    
    # Redirect back to dashboard with success message
    return RedirectResponse(
        url=f"/mentor/dashboard?token={token}&success=Transcript generated and saved successfully", 
        status_code=303
    )

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
            query = "SELECT name, student_email, results, ai_feedback, submission_date FROM results WHERE batch = ?"
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
def get_transcript_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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
    batch: str,
    mode: str = "daily"  # "daily" or "overall"
):
    user = verify_token(token)
    if user["role"] != "student":
        raise HTTPException(status_code=403, detail="Access denied")

    conn = get_db_connection()
    cursor = conn.cursor()
    questions = []
    transcript_text = ""

    try:
        if mode == "daily":
            query = "SELECT TOP 1 transcript FROM transcripts WHERE batch = ? ORDER BY upload_date DESC"
            cursor.execute(query, (batch,))
            record = fetchone_dict(cursor)

            if not record:
                raise HTTPException(status_code=404, detail=f"No transcript found for batch '{batch}'")

            transcript_text = record["transcript"]
            transcript_hash = get_transcript_hash(transcript_text)

            # Check if questions already exist for this transcript
            cursor.execute(
                "SELECT question FROM daily_questions WHERE batch = ? AND transcript_hash = ?",
                (batch, transcript_hash)
            )
            rows = fetchall_dict(cursor)
            if rows:
                questions = generate_questions_with_openai(transcript_text)
                if batch.lower() in ["b1", "b2"]:
                    questions += ask_random_questions()
            else:
                # Generate questions using OpenAI and store them
                questions = generate_questions_with_openai(transcript_text)
                if batch.lower() in ["b1", "b2"]:
                    questions += ask_random_questions()

                for question in questions:
                    cursor.execute(
                        "INSERT INTO daily_questions (batch, question, transcript_hash) VALUES (?, ?, ?)",
                        (batch, question, transcript_hash)
                    )
                conn.commit()

        elif mode == "overall":
            cursor.execute("SELECT question FROM daily_questions WHERE batch = ?", (batch,))
            rows = fetchall_dict(cursor)
            questions = [row["question"] for row in rows]

            random.shuffle(questions) 
            questions = questions[:30]  
            
            if not questions:
                raise HTTPException(status_code=404, detail=f"No stored questions found for batch '{batch}'")

        else:
            raise HTTPException(status_code=400, detail="Invalid mode. Use 'daily' or 'overall'.")

    except pyodbc.Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        cursor.close()
        conn.close()

    return templates.TemplateResponse("quiz.html", {
        "request": request,
        "token": token,
        "user": user,
        "batch": batch,
        "transcript": transcript_text,
        "questions": questions,
        "submitted": False,
        "results": None,
        "mode": mode
    })


@app.post("/student/quiz", response_class=HTMLResponse)
async def post_quiz(
    request: Request,
    token: str = Form(...),
    batch: str = Form(...),
    user_name: str = Form(...),
    questions_json: str = Form(...),
    answers: List[str] = Form(...)
):
    """
    Student submits answers for a quiz, we then evaluate them (using OpenAI),
    store them in DB, and display the feedback on the same page.
    """
    import json

    questions = json.loads(questions_json)

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
    for q, a in zip(questions, answers):
        combined_answers += f"Q: {q}: {a}\n"

    # Evaluate answers using OpenAI
    ai_feedback = evaluate_answers_with_openai(combined_answers, questions)
    
    # Save results
    try:
        insert_query = "INSERT INTO results (name, student_email, batch, results, ai_feedback ) VALUES (?, ?, ?, ?, ?)"
        cursor.execute(insert_query, (user_name, user["email"], batch, combined_answers, ai_feedback))
        conn.commit()
    except pyodbc.Error as e:
        cursor.close()
        conn.close()
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    cursor.close()
    conn.close()

    return templates.TemplateResponse("quiz.html", {
        "request": request,
        "token": token,
        "user": user,
        "batch": batch,
        "transcript": transcript_text,
        "questions": questions,
        "submitted": True,
        "results": {
            "name": user_name,
            "answers": combined_answers,
            "ai_feedback": ai_feedback
        }
    })

