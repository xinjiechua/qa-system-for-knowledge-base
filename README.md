# Question Answering System for a Small Knowledge Base

A question-answering chatbot powered by **Gradio** and built with **Retrieval-Augmented Generation (RAG)**, designed to assist new undergraduate students in quickly querying information from their faculty handbooks for the current academic session.


## Getting Started
Follow these steps to set up and run the application:

### 1. Clone the Repository
```bash
git clone https://github.com/xinjiechua/qa-system-for-knowledge-base
cd qa-system-for-knowledge-base
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

For macOS, install the following system dependencies:
```bash
brew install tesseract
brew install poppler
```

### 3. Configure Environment Variables:
Create a `.env` file based on the provided `.env.example` file and update it with your API keys and other necessary configuration values.

### 4. Start Qdrant Vector Database (via Docker)
Create and run the container for the first time:
```bash
docker run -p 6333:6333 \
-v $(pwd)/vectordb:/qdrant/storage \
qdrant/qdrant
```
Subsequent runs:
```bash
docker ps -a
docker start <CONTAINTER_ID>
```

Note: To ingest data into the vector database for the first time, set `INSERT_DATA = True` in `app.py`. After the initial data ingestion, always set `INSERT_DATA = False`.


### 5. Run the Application
```bash
python app.py
```
Once the application is running, you can access it via your browser at:
http://127.0.0.1:7860
