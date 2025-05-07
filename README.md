# Question Answering System for a Small Knowledge Base

A question-answering chatbot powered by **Gradio** and built with **Retrieval-Augmented Generation (RAG)**, designed to assist new undergraduate students in quickly querying information from their faculty handbooks for the current academic session.


## Getting Started
Follow these steps to set up and run the application:

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

For macOS, `brew install tesseract` and `brew install poppler`

### 2. Set Up Environment Variables:
Create a `.env` file based on the provided `.env.example` file and update it with your API keys and other necessary configuration values.
Get API key at [GEMINI]https://aistudio.google.com/apikey 

### 3. Start Qdrant Vector Database (via Docker)
```bash
docker run -p 6333:6333 \
-v $(pwd)/vectordb:/qdrant/storage \
qdrant/qdrant
```

### 4. Run the Application
```bash
python app.py
```
Once the application is running, you can access it via your browser at:
http://127.0.0.1:7860