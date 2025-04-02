import docx
import random
import re

def extract_questions_from_docx(file_path):
    doc = docx.Document(file_path)
    questions = []

    for para in doc.paragraphs:
        text = para.text.strip()
        # Remove any "Q1:", "Q123." or similar patterns from beginning
        cleaned = re.sub(r"^(Q\d+[:.)\s]*)", "", text)
        if cleaned and "?" in cleaned:
            questions.append(cleaned)

    return questions

def ask_random_questions(docx_file_path, number_of_questions=10):
    questions = extract_questions_from_docx(docx_file_path)
    if not questions:
        print("No questions found in the document.")
        return

    random_questions = random.sample(questions, min(number_of_questions, len(questions)))

    print("\n🎯 Random Interview Questions:\n")
    for i, question in enumerate(random_questions, 1):
        print(f"{i}. {question}")

# 👉 Replace 'your_file.docx' with your actual file name
if __name__ == "__main__":
    docx_file_path = r"C:\Users\DELL 3410\Projects\fast_app\clean_questions.docx"  # ⬅️ Set your actual file name here
    ask_random_questions(docx_file_path, number_of_questions=10)
