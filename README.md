
# 🎓 AI-Based Mentor-Student Evaluation Platform

This is an intelligent web-based evaluation system designed for academic mentors and students. It supports mentor-driven class uploads, AI-powered quiz generation from transcripts, student self-assessments, and automated grading using OpenAI.

---

## 🚀 Features

### 👨‍🏫 Mentor Capabilities:
- Upload class recordings (or fetch from MongoDB).
- Transcribe videos using Whisper AI.
- Automatically generate quiz questions using GPT-4o-mini.
- View batch-wise quiz results with AI-generated feedback.

### 👨‍🎓 Student Capabilities:
- Take daily or overall quizzes.
- Submit answers and receive immediate AI-based feedback.
- View feedback, correct answers, and scores.

### 🧠 AI Capabilities:
- Transcription: OpenAI Whisper (`tiny` and `small` models).
- Question generation: `gpt-4o-mini`.
- Answer evaluation and scoring: GPT-generated detailed feedback.
- Random question selection from a DOCX fallback set.

---

## 🗂 Project Structure

```
├── templates/
│   ├── index.html
│   ├── mentor_dashboard.html
│   ├── mentor_results.html
│   ├── student_dashboard.html
│   └── student_test.html
├── videos/                         # (Optional) for video uploads
├── cert.pem / key.pem             # SSL Certificates (do not share)
├── clean_questions.docx           # Random fallback questions
├── drive.py / sqldb.py            # Utility files
├── main.py                        # Main FastAPI application
├── requirements.txt               # Python dependencies
├── run_fastapi.bat                # Windows shortcut to run server
├── server_log.txt                 # Runtime log output
└── .gitignore
```

---

## ⚙️ Setup Instructions

### 1. Clone the Repo

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Environment Variables

Create a `.env` file or export them manually:

```bash
export OPENAI_API_KEY=your_openai_key
export JWT_SECRET=your_secret_key
```

---

## ▶️ Running the App

### Option 1: Run with `uvicorn`

```bash
uvicorn main:app --reload
```

### Option 2: Use provided batch script (Windows only)

```bash
run_fastapi.bat
```

---

## 🛠 Dependencies

See `requirements.txt`, but major ones include:

- `fastapi`
- `openai`
- `pymongo`
- `pyodbc`
- `whisper`
- `jinja2`
- `python-jose`
- `pillow`, `gridfs`, `docx`

---

## 🔐 Security Notes

- Do **not commit** your `cert.pem`, `key.pem`, or `.docx` with real questions.
- `.gitignore` already includes sensitive folders and files like:
  - `venv/`, `*.pem`, `*.docx`, `videos/`, and `*.log`

---

## 🧪 Sample API Endpoints

- `GET /` — Homepage/login
- `POST /login` — User login (mentor or student)
- `POST /register` — Register new user
- `POST /mentor/upload` — Upload video and generate transcript
- `POST /mentor/fetch_transcript` — Fetch video from MongoDB and transcribe
- `GET/POST /student/student_test` — Start/submit quiz
- `POST /stt_whisper` — Optional: whisper-based STT via upload

---

## 📚 Credits

Built with ❤️ using:
- [FastAPI](https://fastapi.tiangolo.com/)
- [OpenAI](https://openai.com/)
- [MongoDB GridFS](https://www.mongodb.com/docs/manual/core/gridfs/)
- [SQL Server](https://www.microsoft.com/en-us/sql-server)

---

## 📜 License

MIT License. Use at your own risk. Contributions welcome!
```
