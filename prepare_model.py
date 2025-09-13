import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer, util

# 🟢 Load the CSV dataset
df = pd.read_csv("C cube dataset.csv")  # Make sure this CSV is in the same folder

# 🧠 Load the sentence transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# 🔠 Encode all the questions from the CSV
question_embeddings = model.encode(df['Question'].tolist(), convert_to_tensor=True)

# 💾 Save the model, questions, and dataframe to a local .pkl file
model_data = {
    'model': model,
    'question_embeddings': question_embeddings,
    'df': df
}

with open("chatbot_model.pkl", "wb") as f:
    pickle.dump(model_data, f)

print("✅ Model saved locally as chatbot_model.pkl")