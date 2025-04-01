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

# Example usage
questions = ask_random_questions()
print(questions)