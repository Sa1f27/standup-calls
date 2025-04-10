import pyodbc
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from contextlib import asynccontextmanager

# Database configuration
DB_CONFIG = {
    'server': '192.168.48.200',  # SQL Server IP (private)
    'database': 'mentoring_system',
    'username': 'sa',
    'password': 'Welcome@123',
    'driver': '{ODBC Driver 17 for SQL Server}'
}

# SQL scripts for table creation
TABLES_SQL = {
    "users": """
        IF NOT EXISTS (SELECT * FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = 'users')
        BEGIN
            CREATE TABLE users (
                id INT IDENTITY(1,1) PRIMARY KEY,
                email VARCHAR(255) UNIQUE,
                password VARCHAR(255),
                role VARCHAR(50) CHECK (role IN ('mentor', 'student'))
            );
        END
    """,
    "transcripts": """
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
    """,
    "results": """
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
    """
}

# Database connection function
def connect_to_db():
    conn_str = (
        f"DRIVER={DB_CONFIG['driver']};"
        f"SERVER={DB_CONFIG['server']};"
        f"DATABASE={DB_CONFIG['database']};"
        f"UID={DB_CONFIG['username']};"
        f"PWD={DB_CONFIG['password']}"
    )
    return pyodbc.connect(conn_str)

# Ensure tables exist
def ensure_tables_exist():
    try:
        conn = connect_to_db()
        cursor = conn.cursor()
        for table_name, sql in TABLES_SQL.items():
            cursor.execute(sql)
            print(f"Table '{table_name}' ensured/created successfully")
        conn.commit()
    except pyodbc.Error as e:
        print(f"Error creating tables: {str(e)}")
    finally:
        cursor.close()
        conn.close()

# Lifespan handler
@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_tables_exist()
    yield
    print("Application shutdown complete")

# Pydantic models
class User(BaseModel):
    email: str
    password: str
    role: str

class Transcript(BaseModel):
    mentor_email: str
    batch: str
    transcript: str

class Result(BaseModel):
    student_email: str
    batch: str
    results: str
    ai_feedback: Optional[str] = None

# FastAPI app
app = FastAPI(title="Education System API", lifespan=lifespan)

# --- Users Endpoints ---
@app.post("/users/", response_model=dict)
def create_user(user: User):
    try:
        if user.role not in ['mentor', 'student']:
            raise ValueError("Role must be 'mentor' or 'student'")
        conn = connect_to_db()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO users (email, password, role)
            VALUES (?, ?, ?)
        """, (user.email, user.password, user.role))
        conn.commit()
        return {"message": f"User {user.email} created successfully"}
    except pyodbc.Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        cursor.close()
        conn.close()

@app.get("/users/", response_model=List[dict])
def get_all_users():
    try:
        conn = connect_to_db()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users")
        rows = cursor.fetchall()
        return [{"id": row.id, "email": row.email, "role": row.role} for row in rows]  # Exclude password
    except pyodbc.Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        cursor.close()
        conn.close()

@app.get("/users/{user_id}", response_model=dict)
def get_user(user_id: int):
    try:
        conn = connect_to_db()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
        row = cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="User not found")
        return {"id": row.id, "email": row.email, "role": row.role}  # Exclude password
    except pyodbc.Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        cursor.close()
        conn.close()

@app.put("/users/{user_id}", response_model=dict)
def update_user(user_id: int, user: User):
    try:
        if user.role not in ['mentor', 'student']:
            raise ValueError("Role must be 'mentor' or 'student'")
        conn = connect_to_db()
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE users
            SET email = ?, password = ?, role = ?
            WHERE id = ?
        """, (user.email, user.password, user.role, user_id))
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="User not found")
        conn.commit()
        return {"message": f"User {user_id} updated successfully"}
    except pyodbc.Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        cursor.close()
        conn.close()

@app.delete("/users/{user_id}", response_model=dict)
def delete_user(user_id: int):
    try:
        conn = connect_to_db()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="User not found")
        conn.commit()
        return {"message": f"User {user_id} deleted successfully"}
    except pyodbc.Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        cursor.close()
        conn.close()

# --- Transcripts Endpoints ---
@app.post("/transcripts/", response_model=dict)
def create_transcript(transcript: Transcript):
    try:
        conn = connect_to_db()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO transcripts (mentor_email, batch, transcript)
            VALUES (?, ?, ?)
        """, (transcript.mentor_email, transcript.batch, transcript.transcript))
        conn.commit()
        return {"message": f"Transcript for {transcript.mentor_email} created successfully"}
    except pyodbc.Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        cursor.close()
        conn.close()

@app.get("/transcripts/", response_model=List[dict])
def get_all_transcripts():
    try:
        conn = connect_to_db()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM transcripts")
        rows = cursor.fetchall()
        return [{"id": row.id, "mentor_email": row.mentor_email, "batch": row.batch, "transcript": row.transcript, "upload_date": row.upload_date} for row in rows]
    except pyodbc.Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        cursor.close()
        conn.close()

@app.get("/transcripts/{transcript_id}", response_model=dict)
def get_transcript(transcript_id: int):
    try:
        conn = connect_to_db()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM transcripts WHERE id = ?", (transcript_id,))
        row = cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Transcript not found")
        return {"id": row.id, "mentor_email": row.mentor_email, "batch": row.batch, "transcript": row.transcript, "upload_date": row.upload_date}
    except pyodbc.Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        cursor.close()
        conn.close()

@app.put("/transcripts/{transcript_id}", response_model=dict)
def update_transcript(transcript_id: int, transcript: Transcript):
    try:
        conn = connect_to_db()
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE transcripts
            SET mentor_email = ?, batch = ?, transcript = ?
            WHERE id = ?
        """, (transcript.mentor_email, transcript.batch, transcript.transcript, transcript_id))
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Transcript not found")
        conn.commit()
        return {"message": f"Transcript {transcript_id} updated successfully"}
    except pyodbc.Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        cursor.close()
        conn.close()

@app.delete("/transcripts/{transcript_id}", response_model=dict)
def delete_transcript(transcript_id: int):
    try:
        conn = connect_to_db()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM transcripts WHERE id = ?", (transcript_id,))
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Transcript not found")
        conn.commit()
        return {"message": f"Transcript {transcript_id} deleted successfully"}
    except pyodbc.Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        cursor.close()
        conn.close()

# --- Results Endpoints ---
@app.post("/results/", response_model=dict)
def create_result(result: Result):
    try:
        conn = connect_to_db()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO results (student_email, batch, results, ai_feedback)
            VALUES (?, ?, ?, ?)
        """, (result.student_email, result.batch, result.results, result.ai_feedback))
        conn.commit()
        return {"message": f"Result for {result.student_email} created successfully"}
    except pyodbc.Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        cursor.close()
        conn.close()

@app.get("/results/", response_model=List[dict])
def get_all_results():
    try:
        conn = connect_to_db()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM results")
        rows = cursor.fetchall()
        return [{"id": row.id, "student_email": row.student_email, "batch": row.batch, "results": row.results, "ai_feedback": row.ai_feedback, "submission_date": row.submission_date} for row in rows]
    except pyodbc.Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        cursor.close()
        conn.close()

@app.get("/results/{result_id}", response_model=dict)
def get_result(result_id: int):
    try:
        conn = connect_to_db()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM results WHERE id = ?", (result_id,))
        row = cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Result not found")
        return {"id": row.id, "student_email": row.student_email, "batch": row.batch, "results": row.results, "ai_feedback": row.ai_feedback, "submission_date": row.submission_date}
    except pyodbc.Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        cursor.close()
        conn.close()

@app.put("/results/{result_id}", response_model=dict)
def update_result(result_id: int, result: Result):
    try:
        conn = connect_to_db()
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE results
            SET student_email = ?, batch = ?, results = ?, ai_feedback = ?
            WHERE id = ?
        """, (result.student_email, result.batch, result.results, result.ai_feedback, result_id))
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Result not found")
        conn.commit()
        return {"message": f"Result {result_id} updated successfully"}
    except pyodbc.Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        cursor.close()
        conn.close()

@app.delete("/results/{result_id}", response_model=dict)
def delete_result(result_id: int):
    try:
        conn = connect_to_db()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM results WHERE id = ?", (result_id,))
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Result not found")
        conn.commit()
        return {"message": f"Result {result_id} deleted successfully"}
    except pyodbc.Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        cursor.close()
        conn.close()

# Run the app
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)