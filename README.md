# An OpenAI and RAG-Powered Chat bot for Internship Program Assistance

## Overview
This chatbot is designed to assist students at the University of New Hampshire (UNH) with internship-related queries. It dynamically retrieves information from multiple PDF documents using Retrieval-Augmented Generation (RAG) techniques and provides conversational, context-aware responses. The chatbot integrates **Flask**, **LangChain**, **Chroma**, and **OpenAI's GPT-4 Turbo**.

## Key Components
### Preprocessing
- **PDF Extraction**: Extracts text and table data using `pdfplumber`.
- **Chunking**: Splits text into smaller, manageable chunks using `RecursiveCharacterTextSplitter` and custom week-based logic.
- **Embeddings**: Converts text chunks into embeddings using OpenAI's `text-embedding-ada-002`.
- **Storage**: Saves embeddings in Chroma for efficient similarity-based searches.

### Query Handling
- **Input Validation**: Ensures valid input from the user.
- **Cached Response**: Checks if the query has been answered before and retrieves it from the cache.
- **Similarity Search**: Retrieves relevant chunks from the Chroma database.
- **Custom Prompt**: Dynamically formats retrieved content and past chat history into a structured prompt.
- **Response Generation**: Uses GPT-4 Turbo to generate a conversational and context-aware response.

### Post-Processing
- **Memory Update**: Stores the last 5 interactions (questions and answers) in `conversation_memory.json` for handling follow-up questions.
- **Caching**: Saves new questions, responses, and their corresponding hashes in `chat_history.json` to reduce redundant API calls.

## Architecture Diagram
![Chatbot Architecture](static/architecture.png)

## Setup and Installation

### Prerequisites
- Python 3.8+
- OpenAI API Key
- Flask
- LangChain
- Chroma

## Creating an OpenAI API Key
To use OpenAI's GPT-4 API, you need to create an API key from OpenAI's platform.

1. **Sign up or log in to OpenAI**:
   - Visit [OpenAI's website](https://platform.openai.com/signup/) and create an account or log in to your existing account.

2. **Create an API key**:
   - Navigate to the **API Keys** section under your OpenAI dashboard.
   - Click on "Create API Key."
   - Copy the generated API key and store it securely.

## Adding Credits to OpenAI Account
To ensure your API key has sufficient credits for usage:

1. **Log in to OpenAI**:
   - Visit [OpenAI's website](https://platform.openai.com/login) and log in to your account.

2. **Navigate to Billing**:
   - Go to the **Billing** section on the OpenAI dashboard.

3. **Add a payment method**:
   - Add your credit/debit card or other payment methods to your OpenAI account.

4. **Purchase Credits**:
   - In the billing section, choose a payment plan or purchase credits as needed.
   - Ensure you monitor your API usage to avoid exceeding your credits.

5. **Check Remaining Credits**:
   - Under **Usage** on the OpenAI dashboard, you can monitor your credit usage and remaining balance.

---

By following these steps, you can ensure your OpenAI API key is properly set up and has sufficient credits for smooth operation of your chatbot.


### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/UNHM-TEAM-PROJECT/Fall2024-Team-Gulliver.git TeamGulliver
   cd TeamGulliver
2. Create a virtual environment:
   ```bash
   python -m venv venv
   # on Mac/linux: source venv/bin/activate   
   # On Windows: venv\Scripts\activate
3. Install dependencies:
    ```bash
    pip install -r requirements.txt
4. Set your OpenAI API key as an environment variable:
   - **Linux/MacOS**:
     ```bash
     export OPENAI_API_KEY="your-openai-api-key"
     echo $OPENAI_API_KEY  # To verify the key is exported correctly
     ```
   - **Windows**:
     ```cmd
     set OPENAI_API_KEY="your-openai-api-key"
     echo %OPENAI_API_KEY%  # To verify the key is exported correctly
     ```

---

## Usage
1. Place the PDF documents in the `data/` directory.
2. **Change the path to your local directory**:
   - Open the `chatbot.py` file.
   - Update the `pdf_directory ` variable to point to your local directory containing the PDFs. For example:
     ```python
     pdf_directory = "your_local_path_to_pdfs_directory"
     ```
3. Run the chatbot:
   ```bash
   python chatbot.py
4. Open your browser and navigate to:
   ```bash
   http://127.0.0.1:80/
5. Ask internship-related questions, e.g., "What are the requirements for COMP893?"

## Automated Testing

Automated testing scripts are located in the `automated_testing/` folder. These scripts are used to evaluate the chatbot's performance by comparing generated answers with reference answers. They use similarity scoring to calculate accuracy efficiently.

For detailed instructions on setting up and running automated tests, refer to the dedicated `README.md` file located in the `automated_testing/` folder.

### Important Notes:
- The `testing/` folder contains earlier testing scripts that are **not relevant** to the current project. These were created during the initial phases of the sprint and should be ignored.
- The `test_chatbot.py` file is also **not relevant** and can be ignored for current testing purposes.
- For accurate evaluation, focus only on the scripts and instructions provided in the `automated_testing/` folder.


# Deploying Chatbot to AWS

This guide provides step-by-step instructions for deploying applications on Amazon Web Services (AWS). It covers the entire process from account creation to application deployment using EC2 instances.



## Prerequisites
- Basic knowledge of AWS services.
- Installed tools:
  - AWS CLI
  - Python 3.8+
  - Virtual environment tools (e.g., `venv` or `virtualenv`)
  - MobaXTerm or an SSH client for server access.



## Steps to Deploy

### 1. **Create an AWS Account**
1. Go to the [AWS website](https://aws.amazon.com/).
2. Click **"Create an AWS Account"**.
3. Follow the steps to sign up, including:
   - Adding payment information.
   - Verifying your email and phone number.
4. Log in to AWS using your credentials.

### 2. **Launch an EC2 Instance**
1. Go to the AWS Management Console and open the EC2 Dashboard.
2. Click Launch Instance.
3. Configure the instance:
      - Choose an Amazon Machine Image (AMI): Select Amazon Linux 2.
4. Select an instance type: Use t3.2xlarge or similar for performance.

5. Create a new key pair (.pem file) during the instance setup.
6. Download and save the .pem file securely on your local machine. This file will be used for SSH access.
7. Add storage: Allocate at least 100GB.

8. Configure security group to allow the following:
      - Open ports 22 (SSH) and 80 (HTTP).

9. Launch the instance.

### 3. **Start the EC2 Instance**
1. From EC2 Dashboard, select your instance
2. Click "Start Instance"
3. Wait for the instance state to become "Running"
4. Note the Public IPv4 address

### 4. **SSH Connection Setup**
1. Download MobaXterm on windows:
   - Visit the official MobaXterm website: https://mobaxterm.mobatek.net/.
   - Download the "Home Edition" (Installer version or Portable version).
   - Open the downloaded .exe file.
   - Follow the on-screen instructions to install the application.
   - Once installed, open MobaXterm from the Start Menu or Desktop Shortcut.

2. Click "Session" → "New Session"
3. Select "SSH"
4. Configure SSH session:
      - Enter Public IPv4 address in "Remote host"
      - Check "Specify username" and enter "ec2-user"
      - In "Advanced SSH settings", use "Use private key" and select your .pem file
5. Then you will be logged into AWS Linux terminal.

### 5. **Application Deployment**
1. In AWS Linux terminal, switch to root user:
   ```bash
   sudo su
   ```
2. Update system packages:
   ```bash
   sudo yum update -y
   ```
3. Install necessary tools:
   ```bash
   sudo yum install git -y
   sudo yum install python3-pip -y
   ```
4. Clone your repository from Github:
   ```bash
   git clone https://github.com/UNHM-TEAM-PROJECT/Fall2024-Team-Gulliver.git
   cd Fall2024-Team-Gulliver
   ```

5. Install project dependencies:
   ```bash
   pip install -r requirements.txt
   ```
6. Update the chatbot.py file with the following:
   - Navigate to the data folder.
      ```bash
      cd data
      pwd
      ```
   - Then you will get the data directory path where PDFs are located. Copy this path and save it.
7. Now go back to Chatbot.py file and open that file using:
   ```bash
      nano chatbot.py or
      vi chatbot.py
      ```
8. Now chatbot.py file is opened, then go to insert mode typing "i".

9. Find the pdf_directory variable and change the PDF path to copied path in the step 6.
   - pdf_directory  = "Path you have copied from Step 6"

10. Click on Esc button to exit from insert mode and type :wq to save and exit the file.

11. Set the OpenAI API key in the AWS terminal:
      ```bash
      export OPENAI_API_KEY="your_openai_api_key"
      ```
12. Run the Application:
      ```bash
      python3 chatbot.py
      ```

13. Ensure the application is running, and open any browser:
   - Navigate to `http://<public-ip>:5000` in your browser.

   - Start interacting with the chatbot.

