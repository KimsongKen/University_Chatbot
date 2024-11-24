import os
import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import ollama
from PIL import Image
import requests  # Add this import
from io import BytesIO

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How can I help you?"}]
if "data" not in st.session_state:
    st.session_state.data = None
if "context_embeddings" not in st.session_state:
    st.session_state.context_embeddings = None
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

@st.cache_data
def load_and_combine_data(files):
    """Load and combine data from multiple CSV files."""
    combined_data = []
    
    for file_path in files:
        df = pd.read_csv(file_path)
        combined_data.append(df)
    
    combined_df = pd.concat(combined_data, ignore_index=True)
    context_text = combined_df.apply(lambda row: ' | '.join(map(str, row)), axis=1).tolist()
    context_embeddings = st.session_state.model.encode(context_text)
    
    return combined_df, context_embeddings

# File paths for the datasets
file_paths = ['CE.csv', 'Arjans.csv', 'combined_faqs.csv']
st.session_state.data, st.session_state.context_embeddings = load_and_combine_data(file_paths)

st.sidebar.success("Data loaded and combined successfully from all files.")

def search_combined_context(query):
    """Search the combined context embeddings for the most relevant data."""
    if st.session_state.data is None or st.session_state.context_embeddings is None:
        return "I don't have any data loaded to answer your query.", None
    
    user_query_embedding = st.session_state.model.encode([query])
    similarity_scores = util.pytorch_cos_sim(user_query_embedding, st.session_state.context_embeddings)
    
    most_similar_idx = int(similarity_scores.argmax())
    relevant_context = st.session_state.data.iloc[most_similar_idx]
    
    return relevant_context

def handle_special_queries(prompt):
    """Handle queries for all course names, professor names, free elective courses, or ABAC map."""
    if "all course names" in prompt.lower():
        if 'Course_Name' in st.session_state.data.columns:
            course_names = st.session_state.data['Course_Name'].dropna().unique().tolist()
            response = "Here are all the course names:\n" + "\n".join(course_names)
        else:
            response = "Sorry, I couldn't find the course names in the data."
        return response, None

    elif "all professor names" in prompt.lower():
        if 'Professor_Name' in st.session_state.data.columns:
            professor_names = st.session_state.data['Professor_Name'].dropna().unique().tolist()
            response = "Here are all the professor names:\n" + "\n".join(professor_names)
        else:
            response = "Sorry, I couldn't find the professor names in the data."
        return response, None

    elif "free elective" in prompt.lower():
        response = ("For free electives, you can take 2 courses (6 credits) from any faculty. "
                    "Examples include GE1405 (Thai Language and Culture), GE2101 (World Civilization), "
                    "and BG1301 (Business Law 1). Ensure to check prerequisites and course availability "
                    "on the faculty website.")
        return response, None

    elif "map of abac" in prompt.lower():
        response = "Here's the map of ABAC (Assumption University):"
        # Fetch the image from the website
        image_url = "https://admissions.au.edu/wp-content/uploads/2020/10/SUVARNABHUMI-CAMPUS-MAP.jpg"  # Replace with the actual URL of the ABAC map
        try:
            image_response = requests.get(image_url)
            image_response.raise_for_status()  # Raise an exception for bad responses
            image = Image.open(BytesIO(image_response.content))
            return response, image
        except requests.RequestException as e:
            return f"I'm sorry, but I couldn't fetch the ABAC map image from the website. Error: {str(e)}", None

    return None, None

def generate_response(prompt):
    """Generate a response based on the user's prompt."""
    with st.spinner("Generating response..."):
        special_response, image = handle_special_queries(prompt)
        if special_response:
            return special_response, image
        
        relevant_context = search_combined_context(prompt)
        
        context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.conversation_history])
        context += f"\nRelevant context: {relevant_context}"
        
        focused_prompt = f"""Context: {context}
        
        User's question: {prompt}
        
        Please provide a conversational response to the user's question, using the relevant data context."""
        
        llm_response = ollama.chat(model='llama3', messages=[
            {"role": "system", "content": "You are an AI assistant for ABAC university. Respond concisely and accurately."},
            {"role": "user", "content": focused_prompt}
        ])
        response = llm_response['message']['content']
        
        return response, None

def safe_generate_response(prompt):
    """Safely generate a response, handling any errors."""
    try:
        response, image = generate_response(prompt)
        store_user_interaction(prompt, response)
        return response, image
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return "I apologize, but I encountered an error while processing your request. Please try again or contact support if the issue persists.", None

def update_conversation_history(role, content):
    """Update the conversation history with a new message."""
    st.session_state.conversation_history.append({"role": role, "content": content})
    if len(st.session_state.conversation_history) > 10:  # Keep the last 10 messages
        st.session_state.conversation_history.pop(0)

def display_feedback_buttons(message_index):
    """Display feedback buttons for user interaction."""
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

def refresh_page():
    """Refresh the page after each question."""
    st.markdown('<script>window.location.reload();</script>', unsafe_allow_html=True)

# Display chat messages
st.markdown("### Conversation")
for i, msg in enumerate(st.session_state.messages):
    role = "🧑User" if msg["role"] == "user" else "🤖"
    bg_color = "#011f4b" if msg["role"] == "user" else "#1c1c1c"
    alignment = "right" if msg["role"] == "user" else "left"
    
    st.markdown(
        f'<div style="background-color: {bg_color}; color: white; padding: 10px; border-radius: 10px; margin-bottom: 10px; max-width: 80%; float: {alignment};">'
        f'{role}: {msg["content"]}</div>', 
        unsafe_allow_html=True
    )

    if msg["role"] == "assistant" and msg["content"] != "How can I help you?":
        display_feedback_buttons(i)

    # Display image if it exists in the message
    if "image" in msg:
        st.image(msg["image"], caption="ABAC Map", use_column_width=True)

st.markdown('<div style="clear: both;"></div>', unsafe_allow_html=True)
# Handle user input
st.markdown("### Ask a Question")
col1, col2 = st.columns([4, 1])
with col1:
    prompt = st.text_input("Your question:")
with col2:
    if st.button("Send", key="send_button"):
        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            response, image = safe_generate_response(prompt)
            message = {"role": "assistant", "content": response}
            if image:
                message["image"] = image
            st.session_state.messages.append(message)

            st.rerun()  # Add this to refresh the page after a new message is processed

# Add common questions as clickable buttons
st.markdown("### Common Questions")
common_questions = [
    "What is the Map of ABAC?",
    "What are the free elective courses?",
    "What do I do if I miss Ethic Seminar?",
    "Do we have to register English or can we not the English courses?"
]

for question in common_questions:
    if st.button(question):
        prompt = question
        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            response, image = safe_generate_response(prompt)
            message = {"role": "assistant", "content": response}
            if image:
                message["image"] = image
            st.session_state.messages.append(message)

            st.rerun()  # Add this to refresh the page after a new message is processed
 
 #Clear Chat Button
if st.sidebar.button("Clear Chat", key="clear_chat_button"):
    st.session_state.messages = [{"role": "assistant", "content": "How can I help you?"}]
    refresh_page()
