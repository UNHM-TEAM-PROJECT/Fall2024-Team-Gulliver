import os
from flask import Flask, request, jsonify, render_template
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.prompts import PromptTemplate
from langchain.schema import Document
import re
import json
from datetime import datetime

# OpenAI API key and organization
apikey = "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"
organization = "org-ykin6B7UJe9nKDvHiXGf1b9W"

# Initialize Flask app
app = Flask(__name__)
@app.route('/')
def home():
    return render_template('index.html')

# Path to the JSON file for storing chat history
chat_history_file = 'chat_history.json'

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


# Initialize the chat history file if it doesn't exist or is empty
def initialize_chat_history_file():
    if not os.path.exists(chat_history_file) or os.path.getsize(chat_history_file) == 0:
        with open(chat_history_file, 'w') as file:
            json.dump([], file)  # Initialize the file with an empty list

# Load chat history from the JSON file
def load_chat_history():
    if os.path.exists(chat_history_file):
        try:
            with open(chat_history_file, 'r') as file:
                data = file.read().strip()  # Read the file and remove leading/trailing whitespace
                if data:  # Check if the file is not empty
                    return json.loads(data)  # Return parsed JSON
                else:
                    return []  # Return an empty list if the file is empty
        except json.JSONDecodeError:
            # If the file contains invalid JSON, reset it with an empty list
            return []
    return []  # If the file doesn't exist, return an empty list

# Save chat history to the JSON file
def save_chat_history(chat_history):
    with open(chat_history_file, 'w') as file:
        json.dump(chat_history, file, indent=4)

# Append a new chat entry to the JSON file
def save_interaction_to_json(user_question, bot_response):
    chat_history = load_chat_history()  # Load existing chat history
    new_chat = {
        "user_question": user_question,
        "bot_response": bot_response,
        "timestamp": datetime.now().isoformat()
    }
    chat_history.append(new_chat)  # Append the new chat to the history
    save_chat_history(chat_history)  # Save the updated chat history back to the file

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
pdf_directory = '/Users/bubby/TeamGulliver/data'

# Extract text from all PDFs in the directory
documents = extract_texts_from_multiple_pdfs(pdf_directory)

# Step 3: Improve Chunking Strategy for better context retention
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,  # Larger chunk size for more context
    chunk_overlap=200,  # Maintain overlap to preserve context between chunks
    separators=["\n\n", "\n", " "]
)



def custom_split_documents(documents):
    chunks = []
    for doc in documents:
        # Split non-table content and table content separately
        if "<TABLE_START>" in doc.page_content:
            parts = re.split(r"(<TABLE_START>.*?<TABLE_END>)", doc.page_content, flags=re.DOTALL)
            for part in parts:
                if part.startswith("<TABLE_START>"):
                    # Treat the entire table as a single chunk
                    chunks.append(Document(page_content=part, metadata=doc.metadata))
                else:
                    chunked_texts = text_splitter.split_text(part)
                    for chunk in chunked_texts:
                        chunks.append(Document(page_content=chunk, metadata=doc.metadata))
        else:
            chunked_texts = text_splitter.split_text(doc.page_content)
            for chunk in chunked_texts:
                chunks.append(Document(page_content=chunk, metadata=doc.metadata))
    return chunks

# Split documents into chunks with table handling
texts = custom_split_documents(documents)

# Step 4: Load OpenAI Embeddings for Semantic Search
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=apikey)

# Step 5: Create Chroma Index for Vector Store using OpenAI Embeddings
persist_directory = 'db'
if os.path.exists(persist_directory):
    os.system(f"rm -rf {persist_directory}")  # Clear previous vector store

db = Chroma.from_documents(documents=texts, embedding=embeddings, persist_directory=persist_directory)

# Step 6: Create a retriever with OpenAI embeddings
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k":10})  # Retrieve more documents for better accuracy

# Step 7: Create the language model
from langchain.chat_models import ChatOpenAI
llm = ChatOpenAI(
    model="gpt-3.5-turbo",  
    openai_api_key=apikey,
    temperature=0,  # Consistent responses
    top_p=1,        # Consider all tokens
    frequency_penalty=0.0,
    presence_penalty=0.6
)

# Step 8: Create a QA chain with sources
qa = RetrievalQAWithSourcesChain.from_chain_type(llm=llm, retriever=retriever)

# Step 9: Define the prompt template
prompt_template = PromptTemplate(
    input_variables=["question"],
    template="""
    You are an intelligent assistant with access to detailed course syllabi and internship-related documents, including tables.
    Please provide a clear and concise answer to the following question, drawing only from the relevant information in the provided documents.
    Include details from any tables when applicable.

    Question: {question}

    Your answer should be relevant to the user's query, including data from tables when necessary. Ensure to avoid unnecessary elaboration.
    """
)

# Step 10: Validate and process responses
def validate_answer(question, generated_text):
    answer = generated_text.get("answer", "No answer found")

     # First, handle sprint-specific date validation
    if "first sprint end" in question.lower():
        expected_end_date = "October 2nd"
        correct_answer = f"The first sprint ends on {expected_end_date}."
    elif "second sprint end" in question.lower():
        expected_end_date = "November 6th"
        correct_answer = f"The second sprint ends on {expected_end_date}."
    elif "third sprint end" in question.lower():
        expected_end_date = "December 4th"
        correct_answer = f"The third sprint ends on {expected_end_date}."
    else:
        expected_end_date = None
        correct_answer = None
    

    # Check if the generated text contains the correct date for the sprint
    if expected_end_date and expected_end_date not in answer:
        return correct_answer  # Return conversational answer if the end date is wrong
    

    if "scrum master meetings" in question.lower() and "only on mondays" in question.lower():
        correct_meeting_info = ("Scrum master meetings are only on Mondays during "
                                "Week 6, Week 14, and Week 15.")
        if correct_meeting_info not in answer:
            return correct_meeting_info  # Return the correct meeting information


    # Shorten overly verbose answers for concise information
    if len(answer.split()) > 50:  # Limit answer length to 50 words
        answer = ". ".join(answer.split(". ")[:2]) + "."

    
    
    return answer

# Example questions
questions = [
    "do we have scrum meetings only on Wednesdays?",
    "How many sprints are there for the internship project?", 
    "When is the project kickoff?",
    "What are the activities planned in week 2?",
    "How much is the attendance percentage?",
    "How can I register for the internship course?",
    "What are the different internship courses available?",
    "What is the difference between COMP 690 and COMP 693?",
    "How many credits are there for this internship course?",
    "How to make pizza?",
    "What is the building name of the professor is in?",
    "What is cpt?",
    "When is thanks giving break"
]

# Query the Chain and Validate Outputs using `invoke` method
for question in questions:
    query = prompt_template.format(question=question)
    generated_text = qa.invoke({"question": query})  # Use invoke for multiple output keys

    # Extract both the answer and sources
    answer = generated_text.get("answer", "No answer found")
    sources = generated_text.get("sources", "No sources found")

    # Validate and format the output
    print(f"Question: {question}")
    print(f"Answer: {answer}")
    print(f"Sources: {sources}")
    print("-" * 50)

# Flask API for user queries
@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    user_question = data.get('message').strip()

    # Query the model with the user's question
    query = prompt_template.format(question=user_question)
    generated_text = qa.invoke({"question": query})

    # Print relevant chunks retrieved for the user's question
    relevant_docs = retriever.get_relevant_documents(user_question)
    print(f"Documents retrieved for '{user_question}':")
    for doc in relevant_docs:
        print(doc.page_content[:1000])  # Print a portion of the document content

    answer = validate_answer(user_question, generated_text)
    save_interaction_to_json(user_question, answer)
    return jsonify({"response": answer})

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)  # Adjust port if necessary


