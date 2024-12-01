"""

File: chatbot.py
Authors: Shiva Kumar Malyala
Contributors: Hemasri Muddam
Date: 11-30-2024

"""

import os
from flask import Flask, request, jsonify, render_template
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.memory import ConversationBufferMemory
import re
import json
from datetime import datetime
import hashlib
from langchain.chat_models import ChatOpenAI

# Flask app initialization
app = Flask(__name__)

# Route for rendering the homepage
@app.route('/')
def home():
    return render_template('index.html')

# File paths for storing chat history and conversation memory
chat_history_file = 'chat_history.json'
conversation_memory_file = 'conversation_memory.json'

# OpenAI API key for accessing embeddings and language models
apikey = "your_openai_api_key_here"

# Initialize chat history and memory files if they don't exist
def initialize_chat_history_file():
    """Ensures chat history and conversation memory files are available."""
    if not os.path.exists(chat_history_file):
        with open(chat_history_file, 'w') as file:
            json.dump([], file)
    if not os.path.exists(conversation_memory_file):
        with open(conversation_memory_file, 'w') as file:
            json.dump([], file)

# Load general chat history from a JSON file
def load_chat_history():
    """Loads the chat history from a JSON file."""
    if os.path.exists(chat_history_file):
        try:
            with open(chat_history_file, 'r') as file:
                data = json.load(file)
                return data if data else []
        except json.JSONDecodeError:
            return []
    return []

# Save updated chat history to a JSON file
def save_chat_history(chat_history):
    """Writes updated chat history to a JSON file."""
    with open(chat_history_file, 'w') as file:
        json.dump(chat_history, file, indent=4)

# Load conversation memory from a JSON file
def load_conversation_memory():
    """Loads conversation memory for persistent chat context."""
    if os.path.exists(conversation_memory_file):
        try:
            with open(conversation_memory_file, 'r') as file:
                data = json.load(file)
                return data if data else []
        except json.JSONDecodeError:
            return []
    return []

# Save updated conversation memory to a JSON file
def save_conversation_memory(memory):
    """Writes updated conversation memory to a JSON file."""
    with open(conversation_memory_file, 'w') as file:
        json.dump(memory, file, indent=4)

# Generate a hash for a user's question
def get_question_hash(question):
    """Generates a unique hash for the input question to detect duplicates."""
    normalized_question = " ".join(question.lower().split())
    return hashlib.md5(normalized_question.encode()).hexdigest()

# Retrieve cached response for a hashed question
def get_cached_response(question_hash):
    """Checks if a response to the given question hash already exists in history."""
    chat_history = load_chat_history()
    for entry in chat_history:
        if entry.get("question_hash") == question_hash:
            return entry.get("bot_response")
    return None

# Save a user interaction to chat history
def save_interaction_to_json(user_question, bot_response):
    """Stores the user's question and the bot's response in chat history."""
    chat_history = load_chat_history()
    question_hash = get_question_hash(user_question)
    new_chat = {
        "user_question": user_question,
        "bot_response": bot_response,
        "timestamp": datetime.now().isoformat(),
        "question_hash": question_hash
    }
    # Update existing entry or add a new one
    for entry in chat_history:
        if entry.get("question_hash") == question_hash:
            entry.update(new_chat)
            break
    else:
        chat_history.append(new_chat)
    save_chat_history(chat_history)

# Extract text and tables from a single PDF file
def extract_text_from_pdf(pdf_path):
    """Extracts text and table data from a given PDF."""
    text = ''
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ''
            text += page_text + "\n"
            tables = page.extract_table()
            if tables:
                text += "\n\n" + "\n".join(
                    ["\t".join([str(cell) if cell is not None else '' for cell in row]) for row in tables if row]
                ) + "\n"
    return text

# Process multiple PDFs from a directory
def extract_texts_from_multiple_pdfs(pdf_directory):
    """Processes all PDFs in a directory and extracts their content."""
    documents = []
    for pdf_file in os.listdir(pdf_directory):
        if pdf_file.endswith(".pdf"):
            pdf_path = os.path.join(pdf_directory, pdf_file)
            pdf_text = extract_text_from_pdf(pdf_path)
            documents.append(Document(page_content=pdf_text, metadata={"source": pdf_file}))
    return documents

# Directory containing PDFs to be processed
pdf_directory = '/path/to/pdf/directory'
documents = extract_texts_from_multiple_pdfs(pdf_directory)

# Split documents into chunks for better context management
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=200,
    separators=["\n\n", "\n", " "]
)

# Custom document splitting logic based on weekly sections
def custom_split_documents_by_weeks(documents):
    """Splits documents into logical sections by weeks or content."""
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

texts = custom_split_documents_by_weeks(documents)

# Load OpenAI embeddings for vector search
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=apikey)

# Create a Chroma vector store for semantic search
persist_directory = 'db'
if os.path.exists(persist_directory):
    os.system(f"rm -rf {persist_directory}")
db = Chroma.from_documents(documents=texts, embedding=embeddings, persist_directory=persist_directory)

# Configure a retriever for semantic search
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 10})

# Set up conversation memory and load previous context
previous_memory = load_conversation_memory()
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
for item in previous_memory:
    memory.chat_memory.add_user_message(item["user"])
    memory.chat_memory.add_ai_message(item["assistant"])

# Custom prompt template for chatbot
PROMPT_TEMPLATE = """[...truncated for brevity, unchanged from original...]"""

# Initialize the OpenAI chat model
llm = ChatOpenAI(model="gpt-4-turbo", openai_api_key=apikey, temperature=0, top_p=1)

# API route to handle user questions
@app.route('/ask', methods=['POST'])
def ask():
    """Handles user questions and provides AI responses."""
    data = request.get_json()
    user_question = data.get('message', '').strip()

    if not isinstance(user_question, str) or not user_question:
        return jsonify({"error": "Invalid question format."}), 400

    question_hash = get_question_hash(user_question)
    cached_response = get_cached_response(question_hash)
    if cached_response:
        return jsonify({"response": cached_response, "retrieval_context": []})

    relevant_docs = db.similarity_search(user_question, **retriever.search_kwargs)
    retrieval_context = [doc.page_content for doc in relevant_docs]
    context = "\n".join(retrieval_context)

    conversation_memory_data = load_conversation_memory()
    chat_history = "\n".join([
        f"User: {item['user']}\nAssistant: {item['assistant']}" 
        for item in conversation_memory_data
    ])

    prompt_text = PROMPT_TEMPLATE.format(context=context, chat_history=chat_history, question=user_question)
    response = llm.invoke([HumanMessage(content=prompt_text)])
    answer = response.content if response else "No answer found"

    conversation_memory_data.append({"user": user_question, "assistant": answer})
    conversation_memory_data = conversation_memory_data[-5:]
    save_conversation_memory(conversation_memory_data)

    save_interaction_to_json(user_question, answer)

    return jsonify({"response": answer, "retrieval_context": retrieval_context})

if __name__ == '__main__':
    initialize_chat_history_file()
    app.run(debug=True, host='127.0.0.1', port=5000)
