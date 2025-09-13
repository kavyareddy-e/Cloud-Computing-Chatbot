import streamlit as st
import pickle
from sentence_transformers import util

# 🛠 Set Streamlit config FIRST
st.set_page_config(page_title="Cloud Computing Chatbot", page_icon="☁️")

# 🔍 Load model and data
with open("chatbot_model.pkl", "rb") as f:
    data = pickle.load(f)

model = data['model']
question_embeddings = data['question_embeddings']
df = data['df']

# 💬 Get answer using embeddings
def get_answer(user_query, threshold=0.6):
    query_embedding = model.encode(user_query, convert_to_tensor=True)
    similarity_scores = util.cos_sim(query_embedding, question_embeddings)
    best_match_idx = similarity_scores.argmax().item()
    best_score = similarity_scores[0][best_match_idx].item()

    if best_score < threshold:
        return "🤔 Hmm… I’m not sure about that. Try asking something related to cloud computing!"
    
    return df.iloc[best_match_idx]['Answer']

# 🎨 Custom CSS Styling with avatars
st.markdown("""
    <style>
        .chat-container {
            display: flex;
            flex-direction: column;
            gap: 1rem;
            margin-top: 1rem;
        }
        .chat-message {
            display: flex;
            align-items: flex-start;
            gap: 10px;
        }
        .chat-bubble {
            padding: 1rem 1.5rem;
            border-radius: 1.2rem;
            max-width: 70%;
            font-size: 16px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            line-height: 1.4;
            word-wrap: break-word;
        }
        .user-msg {
            align-self: flex-end;
            background: linear-gradient(135deg, #4a90e2, #357ABD);
            color: white;
            border-bottom-right-radius: 0;
        }
        .bot-msg {
            align-self: flex-start;
            background: #EDEAFF;
            color: #333;
            border-bottom-left-radius: 0;
        }
        .avatar {
            width: 36px;
            height: 36px;
            border-radius: 50%;
            object-fit: cover;
        }
        .user-container {
            flex-direction: row-reverse;
        }
    </style>
""", unsafe_allow_html=True)

# 🧠 Title
st.title("☁️ Hey! I'm C-Cube")
st.subheader("Your Mini Cloud Computing Assistant!")

# 🧾 Session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# 🧹 Clear Chat Button
if st.button("🧹 Clear Chat"):
    st.session_state.chat_history = []

# 📥 Input form with auto-clear
with st.form(key='chat_form', clear_on_submit=True):
    user_input = st.text_input("Type your question:")
    submit = st.form_submit_button("Send")

# 🧠 Process input
if submit and user_input:
    answer = get_answer(user_input)
    st.session_state.chat_history.append(("user", user_input))
    st.session_state.chat_history.append(("bot", answer))

# 🖼️ Display chat history with avatars
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

for sender, msg in st.session_state.chat_history:
    avatar_url = (
        "https://cdn-icons-png.flaticon.com/512/1077/1077012.png"
        if sender == "user"
        else "https://cdn-icons-png.flaticon.com/512/4712/4712037.png"
    )
    alignment_class = "user-container" if sender == "user" else ""
    bubble_class = "user-msg" if sender == "user" else "bot-msg"

    st.markdown(f"""
        <div class="chat-message {alignment_class}">
            <img src="{avatar_url}" class="avatar">
            <div class="chat-bubble {bubble_class}">{msg}</div>
        </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)