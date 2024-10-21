#chatbot.py
import os
from flask import Flask, request, jsonify, render_template
import pdfplumber
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.prompts import PromptTemplate
from langchain.schema import Document
import re

# OpenAI API key and organization
apikey = "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"
organization = "org-ykin6B7UJe9nKDvHiXGf1b9W"

# Initialize Flask app
app = Flask(__name__)

# Step 1: Extract text and table data from PDF using pdfplumber
def extract_text_from_pdf(pdf_path):
    text = ''
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ''  # Extract text from page
            text += page_text + "\n"  # Add newline for separation
            tables = page.extract_table()
            if tables:
                # Replace None values with empty strings and format table data
                text += "\n\n" + "\n".join(
                    ["\t".join([str(cell) if cell is not None else '' for cell in row]) for row in tables if row]
                ) + "\n"
    return text

# Step 2: Handle multiple PDFs in a directory
def extract_texts_from_multiple_pdfs(pdf_directory):
    documents = []
    for pdf_file in os.listdir(pdf_directory):
        if pdf_file.endswith(".pdf"):
            pdf_path = os.path.join(pdf_directory, pdf_file)
            pdf_text = extract_text_from_pdf(pdf_path)
            documents.append(Document(page_content=pdf_text, metadata={"source": pdf_file}))
    return documents

# Directory containing your PDF files
pdf_directory = 'C:\\Users\\sindh\\TeamGulliver\\Data\\'

# Extract text from all PDFs in the directory
documents = extract_texts_from_multiple_pdfs(pdf_directory)

# Step 3: Improve Chunking Strategy for better retrieval from all PDFs
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # Small enough for specific information extraction
    chunk_overlap=150,  # Increased overlap for better context retention
    separators=["\n\n", "\n", " "],  # Split based on paragraphs, new lines, or spaces
)
texts = text_splitter.split_documents(documents)

# Step 4: Load OpenAI Embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=apikey)

# Step 5: Create Chroma Index for Vector Store using OpenAI Embeddings
persist_directory = 'db'
if os.path.exists(persist_directory):
    # Clear previous vector store if switching models
    os.system(f"rm -rf {persist_directory}")

db = Chroma.from_documents(documents=texts, embedding=embeddings, persist_directory=persist_directory)

# Step 6: Create a retriever and chain it with OpenAI LLM
retriever = db.as_retriever()

from langchain.llms import OpenAI

llm = OpenAI(
    openai_api_key=apikey,
    openai_organization=organization,
    temperature=0.7,  # Adjust as needed
    top_p=0.9,        # Adjust to control output randomness
    frequency_penalty=0.0,  # Penalty for word repetition
    presence_penalty=0.6    # Encourage new topic introduction
)

# Step 7: Create a QA chain with sources
qa = RetrievalQAWithSourcesChain.from_chain_type(llm=llm, retriever=retriever)

# Step 8: Refine the Prompt Template for More Detailed Answers
prompt_template = PromptTemplate(
    input_variables=["question"],
    template="""
    You are a knowledgeable assistant with access to multiple course syllabi. Answer the following question using specific details from the content.

    Question: {question}

    Ensure to include all relevant information, especially when asked about learning outcomes, planned activities, or instructor details.
    """
)

# Step 9: Validate and Extract Credits or Numerical Values
def extract_numerical_answer(answer):
    match = re.search(r'\d+', answer)
    return match.group(0) if match else "Invalid number format"

def validate_answer(question, generated_text):
    try:
        answer = generated_text.get("answer", "No answer found")


        # Validate numerical answers
        if "credits" in question.lower() or "how many" in question.lower():
            answer = extract_numerical_answer(answer)

    except (IndexError, KeyError, AttributeError):
        return f"Answer not available"

    return f"{answer}"

# Step 10: Query the Chain and Validate Outputs
questions = [
    "When does the first sprint start?",
    "Who is the instructor for the course?",
    "What are the course office hours?",
    "How many credits is the course?",
    "What are the course's learning outcomes?",
    "What are the activities planned for Week 3?",
    "How many sprints are there?",
    "What is the grading policy?"
]

for question in questions:
    query = prompt_template.format(question=question)
    generated_text = qa(query)
    print(f"Question: {question}")
    print(validate_answer(question, generated_text))
    print("-" * 50)

# Create the Flask route to serve the chat interface
@app.route('/')
def home():
    return render_template('index.html')

# Create an API endpoint for handling user queries
@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    user_question = data.get('message')
    query = prompt_template.format(question=user_question)
    generated_text = qa(query)

    answer = validate_answer(user_question, generated_text)
    return jsonify({"response": answer})

if __name__ == '__main__':
    app.run(debug=True)
