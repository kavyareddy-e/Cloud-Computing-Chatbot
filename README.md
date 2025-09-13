# C-Cube: Cloud Computing Chatbot  

This project is an interactive chatbot built with **Streamlit** and **Sentence Transformers** to answer questions related to **Cloud Computing**.  
It works by computing semantic similarity between user queries and a curated dataset of questions and answers.  


## Features
- Semantic search using the `all-MiniLM-L6-v2` model  
- Chat interface built with Streamlit  
- Q&A dataset-driven responses  
- Maintains chat history during a session  



## Project Structure
├── chatbot_app.py # Streamlit chatbot application
├── prepare_model.py # Script to generate embeddings and pickle file
├── C cube dataset.csv # Dataset (Q&A)
├── chatbot_model.pkl # Saved model and embeddings
└── .gitignore # Files and folders to ignore



## Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/your-username/c-cube-chatbot.git
cd c-cube-chatbot
```

### 2. Set up the environment
python -m venv venv
.\venv\Scripts\activate   # On Windows
source venv/bin/activate  # On Mac/Linux

pip install -r requirements.txt

### 3. Generate the model (if chatbot_model.pkl is missing)
python prepare_model.py

### 4. Run the application
streamlit run chatbot_app.py




