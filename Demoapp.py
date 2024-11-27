import os
import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import ollama

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How can I help you?"}]
if "data" not in st.session_state:
    st.session_state.data = {}
if "faq_embeddings" not in st.session_state:
    st.session_state.faq_embeddings = {}
if "model" not in st.session_state:
    st.session_state.model = None
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# Load the Sentence Transformer model once
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

st.session_state.model = load_model()

def collect_feedback(message_index, feedback):
    """Store feedback provided by users."""
    with open('feedback.txt', 'a') as f:
        message = st.session_state.messages[message_index]['content']
        f.write(f"Message: {message}\nFeedback: {feedback}\n\n")

def store_user_interaction(user_question, assistant_response):
    """Store user questions and assistant responses."""
    with open('interactions.txt', 'a') as f:
        f.write(f"User: {user_question}\nAssistant: {assistant_response}\n\n")

def load_data_ce(file_path):
    """Load and process the CE.csv file."""
    df = pd.read_csv(file_path)
    # Transform or prepare the data specifically for CE.csv
    context_text = df.apply(lambda row: ' | '.join(map(str, row)), axis=1).tolist()
    context_embeddings = st.session_state.model.encode(context_text)
    st.session_state.data['CE'] = {'data': df, 'embeddings': context_embeddings}

def load_data_arjans(file_path):
    """Load and process the Arjans.csv file."""
    df = pd.read_csv(file_path)
    # Transform or prepare the data specifically for Arjans.csv
    context_text = df.apply(lambda row: ' | '.join(map(str, row)), axis=1).tolist()
    context_embeddings = st.session_state.model.encode(context_text)
    st.session_state.data['Arjans'] = {'data': df, 'embeddings': context_embeddings}

def load_data_combined_faqs(file_path):
    """Load and process the combined_faqs.csv file."""
    df = pd.read_csv(file_path)
    # Transform or prepare the data specifically for combined_faqs.csv
    context_text = df.apply(lambda row: ' | '.join(map(str, row)), axis=1).tolist()
    context_embeddings = st.session_state.model.encode(context_text)
    st.session_state.data['FAQs'] = {'data': df, 'embeddings': context_embeddings}

# File paths for the three datasets
load_data_ce('CE.csv')
load_data_arjans('Arjans.csv')
load_data_combined_faqs('combined_faqs.csv')  # You need to provide this file path

st.sidebar.success("Data loaded successfully from all files.")

def search_context(query, dataset_key):
    """Search the context embeddings for the most relevant data in a specific dataset."""
    data_info = st.session_state.data.get(dataset_key)
    
    if not data_info:
        return "I don't have any data loaded to answer your query.", None

    data, context_embeddings = data_info['data'], data_info['embeddings']
    user_query_embedding = st.session_state.model.encode([query])
    similarity_scores = util.pytorch_cos_sim(user_query_embedding, context_embeddings)
    most_similar_idx = int(similarity_scores.argmax())
    relevant_context = data.iloc[most_similar_idx]  # Retrieve the most relevant row

    return relevant_context

def generate_response(prompt, dataset_key):
    with st.spinner("Generating response..."):
        relevant_context = search_context(prompt, dataset_key)
        
        # Prepare the context for the model
        context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.conversation_history])
        context += f"\nRelevant context: {relevant_context}"
        
        # Generate a response using the relevant context
        focused_prompt = f"""Context: {context}
        
        User's question: {prompt}
        
        Please provide a conversational response to the user's question, using the relevant data context."""
        
        llm_response = ollama.chat(model='llama3', messages=[
            {"role": "system", "content": "You are an AI assistant for ABAC university. Respond concisely and accurately."},
            {"role": "user", "content": focused_prompt}
        ])
        response = llm_response['message']['content']
        
        return response

def safe_generate_response(prompt, dataset_key):
    try:
        response = generate_response(prompt, dataset_key)
        store_user_interaction(prompt, response)
        return response
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return "I apologize, but I encountered an error while processing your request. Please try again or contact support if the issue persists."

def update_conversation_history(role, content):
    st.session_state.conversation_history.append({"role": role, "content": content})
    if len(st.session_state.conversation_history) > 10:  # Keep last 10 messages
        st.session_state.conversation_history.pop(0)

def display_feedback_buttons(message_index):
    col1, col2 = st.columns(2)
    with col1:
        if st.button("👍", key=f"thumbs_up_{message_index}"):
            collect_feedback(message_index, "Positive")
            st.success("Thank you for your feedback!")
    with col2:
        if st.button("👎", key=f"thumbs_down_{message_index}"):
            feedback = st.text_input("Please provide details on how we can improve:", key=f"feedback_{message_index}")
            if st.button("Submit Feedback", key=f"submit_feedback_{message_index}"):
                collect_feedback(message_index, feedback)
                st.success("Thank you for your feedback!")

# Display chat messages
st.markdown("### Conversation")
for i, msg in enumerate(st.session_state.messages):
    role = "🧑User" if msg["role"] == "user" else "🤖"
    bg_color = "#011f4b" if msg["role"] == "user" else "#1c1c1c"
    alignment = "right" if msg["role"] == "user" else "left"
    st.markdown(f'<div style="background-color: {bg_color}; padding: 10px; border-radius: 10px; margin-bottom: 10px; max-width: 80%; float: {alignment};">{role}: {msg["content"]}</div>', unsafe_allow_html=True)

    if msg["role"] == "assistant" and msg["content"] != "How can I help you?":
        display_feedback_buttons(i)

st.markdown('<div style="clear: both;"></div>', unsafe_allow_html=True)

# Handle user input
st.markdown("### Ask a Question")
prompt = st.text_input("Your question:")

# Dropdown to select dataset
dataset_key = st.selectbox("Select Dataset:", options=["CE", "Arjans", "FAQs"], index=0)

if st.button("Send", key="send_button"):
    if prompt:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Generate and add assistant's response to chat history
        response = safe_generate_response(prompt, dataset_key)
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Write the updated chat history directly
        st.write("### Conversation")
        for i, msg in enumerate(st.session_state.messages):
            role = "🧑User" if msg["role"] == "user" else "🤖"
            bg_color = "#011f4b" if msg["role"] == "user" else "#1c1c1c"
            alignment = "right" if msg["role"] == "user" else "left"
            st.markdown(f'<div style="background-color: {bg_color}; padding: 10px; border-radius: 10px; margin-bottom: 10px; max-width: 80%; float: {alignment};">{role}: {msg["content"]}</div>', unsafe_allow_html=True)

        st.markdown('<div style="clear: both;"></div>', unsafe_allow_html=True)

# Clear Chat Button
if st.sidebar.button("Clear Chat", key="clear_chat_button"):
    st.session_state.messages = [{"role": "assistant", "content": "How can I help you?"}]
    st.experimental_rerun()
