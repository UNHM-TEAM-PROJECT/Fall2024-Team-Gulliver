import os
from flask import Flask, request, jsonify, render_template
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import re
import json
from datetime import datetime
import hashlib

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')


# Path to the JSON file for storing chat history and conversation memory
chat_history_file = 'chat_history.json'
conversation_memory_file = 'conversation_memory.json'

# OpenAI API key and organization (move these to a .env file for security in production)
apikey = "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"

# Initialize the chat history and conversation memory files if they don't exist
def initialize_chat_history_file():
    if not os.path.exists(chat_history_file):
        with open(chat_history_file, 'w') as file:
            json.dump([], file)
    if not os.path.exists(conversation_memory_file):
        with open(conversation_memory_file, 'w') as file:
            json.dump([], file)

# Load general chat history from a JSON file
def load_chat_history():
    if os.path.exists(chat_history_file):
        try:
            with open(chat_history_file, 'r') as file:
                data = json.load(file)
                return data if data else []
        except json.JSONDecodeError:
            return []
    return []

# Save general chat history to a JSON file
def save_chat_history(chat_history):
    with open(chat_history_file, 'w') as file:
        json.dump(chat_history, file, indent=4)

# Load conversation memory from a JSON file for persistent memory across sessions
def load_conversation_memory():
    if os.path.exists(conversation_memory_file):
        try:
            with open(conversation_memory_file, 'r') as file:
                data = json.load(file)
                return data if data else []
        except json.JSONDecodeError:
            return []
    return []

# Save conversation memory to a JSON file after each interaction
def save_conversation_memory(memory):
    with open(conversation_memory_file, 'w') as file:
        json.dump(memory, file, indent=4)

# Generate a consistent hash for a question
def get_question_hash(question):
    normalized_question = " ".join(question.lower().split())
    return hashlib.md5(normalized_question.encode()).hexdigest()

# Retrieve a cached response based on question hash
def get_cached_response(question_hash):
    chat_history = load_chat_history()
    for entry in chat_history:
        if entry.get("question_hash") == question_hash:
            return entry.get("bot_response")
    return None

# Save a new interaction to the chat history with question hash
def save_interaction_to_json(user_question, bot_response):
    chat_history = load_chat_history()
    question_hash = get_question_hash(user_question)
    new_chat = {
        "user_question": user_question,
        "bot_response": bot_response,
        "timestamp": datetime.now().isoformat(),
        "question_hash": question_hash
    }
    # Check if question exists and update if necessary
    for entry in chat_history:
        if entry.get("question_hash") == question_hash:
            entry.update(new_chat)
            break
    else:
        chat_history.append(new_chat)
    
    save_chat_history(chat_history)

# Step 1: Extract text and table data from PDF using pdfplumber
def extract_text_from_pdf(pdf_path):
    text = ''
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ''  # Extract text from page
            text += page_text + "\n"  # Add newline for separation
            tables = page.extract_table()
            if tables:
                text += "\n\n" + "\n".join(
                    ["\t".join([str(cell) if cell is not None else '' for cell in row]) for row in tables if row]
                ) + "\n"
    return text

# Handle multiple PDFs in a directory
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
documents = extract_texts_from_multiple_pdfs(pdf_directory)

# Step 3: Improve Chunking Strategy for better context retention
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=200,
    separators=["\n\n", "\n", " "]
)

# Custom function to split documents based on weeks
def custom_split_documents_by_weeks(documents):
    chunks = []
    for doc in documents:
        if "<TABLE_START>" in doc.page_content:
            week_sections = re.split(r"(Week \d+)", doc.page_content)
            current_week = None
            
            for part in week_sections:
                week_match = re.match(r"Week \d+", part)
                if week_match:
                    current_week = part.strip()
                elif current_week:
                    chunks.append(Document(page_content=f"{current_week}\n{part.strip()}", metadata=doc.metadata))
        else:
            chunked_texts = text_splitter.split_text(doc.page_content)
            for chunk in chunked_texts:
                chunks.append(Document(page_content=chunk, metadata=doc.metadata))
    return chunks

# Split documents into chunks with table handling by weeks
texts = custom_split_documents_by_weeks(documents)

# Step 4: Load OpenAI Embeddings for Semantic Search
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=apikey)

# Step 5: Create Chroma Index for Vector Store using OpenAI Embeddings
persist_directory = 'db'
if os.path.exists(persist_directory):
    os.system(f"rm -rf {persist_directory}")

db = Chroma.from_documents(documents=texts, embedding=embeddings, persist_directory=persist_directory)

# Step 6: Create a retriever with OpenAI embeddings
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k":10})

# Step 7: Set up conversational memory and load past memory
previous_memory = load_conversation_memory()
memory = ConversationBufferMemory(
    memory_key="chat_history", 
    return_messages=True
)
for item in previous_memory:
    # Load past messages into memory
    memory.chat_memory.add_user_message(item["user"])
    memory.chat_memory.add_ai_message(item["assistant"])

# Step 8: Create a Conversational Retrieval Chain with memory
from langchain.chat_models import ChatOpenAI
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    openai_api_key=apikey,
    temperature=0,
    top_p=1,
    frequency_penalty=0.0,
    presence_penalty=0.6
)

conversation_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory
)

# Flask API for user queries with caching and persistent conversational memory
@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    user_question = data.get('message').strip()
    question_hash = get_question_hash(user_question)
    
    # Try retrieving cached response first for identical questions
    cached_response = get_cached_response(question_hash)
    if cached_response:
        return jsonify({"response": cached_response})

    # Generate a new response if no cache is found, leveraging conversation memory
    generated_text = conversation_chain.invoke({"question": user_question})
    
    # Extract answer and save conversation to memory file
    answer = generated_text.get("answer", "No answer found")
    save_interaction_to_json(user_question, answer)

    # Save the current memory state to persistent storage
    memory_history = [
    {"user": m.content, "assistant": r.content} 
    for m, r in zip(memory.load_memory_variables({})["chat_history"][::2], memory.load_memory_variables({})["chat_history"][1::2])
    ]

    save_conversation_memory(memory_history)

    return jsonify({"response": answer})

# Initialize chat history file and run the app
if __name__ == '__main__':
    initialize_chat_history_file()
    app.run(debug=True, host='127.0.0.1', port=5000)

