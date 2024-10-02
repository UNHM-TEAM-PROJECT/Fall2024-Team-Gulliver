# Chatbot.py
import os
import torch
from flask import Flask, request, jsonify, render_template
import pdfplumber
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
import re

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

# Path to the PDF file
pdf_path = '/Users/bubby/Gulliver/data/internship_course.pdf'

pdf_text = extract_text_from_pdf(pdf_path)

# Print the first 1000 characters to check content extraction
print("Extracted Text from PDF (first 1000 chars):", pdf_text[:1000])

# Step 2: Wrap the extracted text into a LangChain Document object
documents = [Document(page_content=pdf_text, metadata={"source": pdf_path})]

# Step 3: Improve Chunking Strategy for Schedule and Learning Outcomes
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # Small enough for specific information extraction
    chunk_overlap=150,  # Increased overlap for better context retention
    separators=["\n\n", "\n", " "],  # Split based on paragraphs, new lines, or spaces
)
texts = text_splitter.split_documents(documents)

# Step 4: Load Embeddings using HuggingFaceEmbeddings
embedding_model_name = 'sentence-transformers/all-MiniLM-L6-v2'
sentence_embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

# Step 5: Load LLM for response generation
LLM_Model = "google/flan-t5-large"
tokenizer = AutoTokenizer.from_pretrained(LLM_Model)
model = AutoModelForSeq2SeqLM.from_pretrained(LLM_Model, torch_dtype=torch.float32)

pipe = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=512,
    temperature=0.7,  
    top_p=0.9,
    repetition_penalty=1.15
)

llm = HuggingFacePipeline(pipeline=pipe)

# Step 6: Create Chroma Index for Vector Store
persist_directory = 'db'
if os.path.exists(persist_directory):
    # Clear previous vector store if switching models
    os.system(f"rm -rf {persist_directory}")

db = Chroma.from_documents(documents=texts, 
                           embedding=sentence_embeddings,
                           persist_directory=persist_directory)

# Step 7: Create a retriever and chain it with LLM
retriever = db.as_retriever(search_kwargs={"k": 3})

# Step 8: Refine the Prompt Template for More Detailed Answers
prompt_template = PromptTemplate(
    input_variables=["question"],
    template="""
    You are a knowledgeable assistant with access to a course syllabus. Answer the following question using specific details from the syllabus content.

    Question: {question}

    Ensure to include all relevant information, especially when asked about learning outcomes or planned activities.
    """
)

qa = RetrievalQA.from_chain_type(llm=llm, 
                                  chain_type="stuff", 
                                  retriever=retriever, 
                                  return_source_documents=True)

# Step 9: Validate and Extract Credits or Numerical Values
def extract_numerical_answer(answer):
    match = re.search(r'\d+', answer)
    return match.group(0) if match else "Invalid number format"

def validate_answer(question, generated_text):
    try:
        answer = generated_text.get("result", "No answer found")
        source = generated_text['source_documents'][0].metadata['source']
        
        # Validate numerical answers
        if "credits" in question.lower() or "how many" in question.lower():
            answer = extract_numerical_answer(answer)
        
    except (IndexError, KeyError, AttributeError):
        return f"Answer: Not available\nSource: Not available"

    return f"Answer: {answer}\nSource: {source}"

#Step 10: Query the Chain and Validate Outputs
questions = [
    "When does the first sprint start?",
    "Who is the instructor for the course?",
    "What are the course office hours?",
    "How many credits is the course?",
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
