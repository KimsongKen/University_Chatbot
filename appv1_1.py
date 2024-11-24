import os
import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util, InputExample, losses
from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from datasets import Dataset, DatasetDict
import numpy as np
from typing import List
import pickle  # For caching
import ollama
from PIL import Image
import requests  # Add this import
from io import BytesIO

# Main Page Layout
icon_path = "abac_logo.jpg"  # Replace with the path to your local image file

col1, col2 = st.columns([1, 5])  # Adjust the ratio for spacing

with col1:
    st.image(icon_path, width=100)  # Adjust the width as needed

with col2:
    st.title("VME Chatbot")

st.markdown("### Your AI-powered assistant for answering questions about ABAC")

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

# Base Embedding Class
class BaseEmbeddingFunction:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        embeddings = self.model.encode(texts, convert_to_tensor=False)
        return np.array(embeddings)

# Initialize the embedding function
embedding_function = BaseEmbeddingFunction()

# Function to prepare datasets
def prepare_datasets(file_paths):
    datasets = {}
    for file_path in file_paths:
        try:
            # Attempt to read the CSV file with UTF-8 encoding
            df = pd.read_csv(file_path, encoding='utf-8')
        except UnicodeDecodeError:
            # If UTF-8 fails, try reading with ISO-8859-1 encoding
            df = pd.read_csv(file_path, encoding='ISO-8859-1')
        
        # Print the column names for debugging
        print(f"Columns in {file_path}: {df.columns.tolist()}")
        
        # Convert DataFrame to Dataset instance
        datasets[file_path] = Dataset.from_pandas(df)
    return DatasetDict(datasets)

# Load or create embeddings for the dataset
def load_or_create_embeddings(file_paths):
    embeddings_cache_file = 'embeddings_cache.pkl'
    
    if os.path.exists(embeddings_cache_file):
        with open(embeddings_cache_file, 'rb') as f:
            context_embeddings = pickle.load(f)
    else:
        datasets = prepare_datasets(file_paths)
        context_texts = []
        
        # Extract relevant text from combined_faqs.csv
        faq_dataset = datasets['combined_faqs.csv']
        for index in range(len(faq_dataset)):
            context_texts.append(faq_dataset[index]['Question'])  # Use 'Question' for FAQs
        
        # Extract relevant text from CE.csv
        ce_dataset = datasets['CE.csv']
        for index in range(len(ce_dataset)):
            context_texts.append(ce_dataset[index]['Course_Name'])  # Use 'Course_Name' for courses
        
        # Extract relevant text from Arjans.csv
        arjans_dataset = datasets['Arjans.csv']
        for index in range(len(arjans_dataset)):
            context_texts.append(arjans_dataset[index]['Professor_Name'])  # Use 'Professor_Name' for professors
        
        context_embeddings = embedding_function.create_embeddings(context_texts)
        
        with open(embeddings_cache_file, 'wb') as f:
            pickle.dump(context_embeddings, f)
    
    return context_embeddings

# File paths for the datasets
file_paths = ['CE.csv', 'Arjans.csv', 'combined_faqs.csv']

# Load or create embeddings for the datasets
context_embeddings = load_or_create_embeddings(file_paths)

def search_combined_context(query):
    """Search the combined context embeddings for the most relevant data."""
    if st.session_state.data is None or st.session_state.context_embeddings is None:
        return "I don't have any data loaded to answer your query.", None
    
    user_query_embedding = st.session_state.model.encode([query])
    similarity_scores = util.pytorch_cos_sim(user_query_embedding, context_embeddings)
    
    most_similar_idx = int(similarity_scores.argmax())
    relevant_context = context_texts[most_similar_idx]  # Get the relevant context
    
    return relevant_context

def handle_special_queries(prompt):
    """Handle general queries, course names, professor names, free electives, or ABAC map."""
    
    # General AI-related queries (make sure to handle non-course-specific questions first)
    if "what are you" in prompt.lower() or "who are you" in prompt.lower():
        response = "I'm an AI assistant for ABAC University, here to help you with any academic information or queries about the university. How can I assist you today?"
        return response, None

    # Course-related queries
    if "course" in prompt.lower() or "subject" in prompt.lower():
        if "all course names" in prompt.lower():
            if 'Course_Name' in st.session_state.data.columns:
                course_names = st.session_state.data['Course_Name'].dropna().unique().tolist()
                response = "Here are all the course names:\n" + "\n".join(course_names)
            else:
                response = "Sorry, I couldn't find the course names in the data."
            return response, None
        elif "free elective" in prompt.lower():
            response = ("For free electives, you can take 2 courses (6 credits) from any faculty. "
                        "Examples include GE1405 (Thai Language and Culture), GE2101 (World Civilization), "
                        "and BG1301 (Business Law 1). Ensure to check prerequisites and course availability "
                        "on the faculty website.")
            return response, None

    # Professor-related queries
    if "professor" in prompt.lower() or "instructor" in prompt.lower():
        if "all professor names" in prompt.lower():
            if 'Professor_Name' in st.session_state.data.columns:
                professor_names = st.session_state.data['Professor_Name'].dropna().unique().tolist()
                response = "Here are all the professor names:\n" + "\n".join(professor_names)
            else:
                response = "Sorry, I couldn't find the professor names in the data."
            return response, None

    # General ABAC questions (e.g., about the campus)
    if "map of abac" in prompt.lower() or "campus" in prompt.lower():
        response = "Here's the map of ABAC (Assumption University):"
        image_url = "https://admissions.au.edu/wp-content/uploads/2020/10/SUVARNABHUMI-CAMPUS-MAP.jpg"
        try:
            image_response = requests.get(image_url)
            image_response.raise_for_status()  
            image = Image.open(BytesIO(image_response.content))
            return response, image
        except requests.RequestException as e:
            return f"I'm sorry, but I couldn't fetch the ABAC map image from the website. Error: {str(e)}", None
    
    # FAQ-based queries (from combined_faqs.csv)
    if "faq" in prompt.lower() or "frequently asked questions" in prompt.lower() or "question" in prompt.lower():
        if 'Question' in st.session_state.data.columns and 'Answer' in st.session_state.data.columns:
            matching_faqs = st.session_state.data[
                st.session_state.data['Question'].str.contains(prompt, case=False, na=False)
            ]
            if not matching_faqs.empty:
                faq_answer = matching_faqs.iloc[0]['Answer']
                response = f"Here's an answer from the FAQs:\n{faq_answer}"
            else:
                response = "Sorry, I couldn't find a relevant FAQ for your query."
            return response, None

    return None, None


def generate_response(prompt):
    """Generate a response based on the user's prompt."""
    with st.spinner("Generating response..."):
        # Check for course or general ABAC queries first
        special_response, image = handle_special_queries(prompt)
        if special_response:
            return special_response, image
        
        # If not a special query, search through the context embeddings
        user_query_embedding = embedding_function.create_embeddings([prompt])
        similarity_scores = util.pytorch_cos_sim(user_query_embedding, context_embeddings)
        
        most_similar_idx = int(similarity_scores.argmax())
        relevant_context = context_texts[most_similar_idx]  # Get the relevant context
        
        context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.conversation_history])
        context += f"\nRelevant context: {relevant_context}"
        
        focused_prompt = f"""Context: {context}
        
        User's question: {prompt}
        
        Please provide a conversational response to the user's question, using the relevant data context."""
        
        # Generate response with ollama
        llm_response = ollama.chat(model='llama3', messages=[
            {"role": "system", "content": "You are an AI assistant for ABAC university. Respond concisely and accurately."},
            {"role": "user", "content": focused_prompt}
        ])
        response = llm_response['message']['content']
        
        # Show preview of the response
        st.markdown("### Preview of the Response:")
        st.write(response)  # Display the generated response as a preview
        
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
    "Do we have to register English or can we not take English courses?"
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