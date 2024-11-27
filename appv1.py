import os
import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import ollama
from PIL import Image
import requests
from io import BytesIO
from typing import Tuple, Optional, List, Dict, Any
import logging
import chardet

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='app.log'
)
logger = logging.getLogger(__name__)

class EmbeddingModel:
    """Handles text embeddings using SentenceTransformer."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts into embeddings."""
        return self.model.encode(texts, convert_to_tensor=True)

class DataLoader:
    """Handles data loading and preprocessing."""
    
    @staticmethod
    def detect_encoding(file_path: str) -> str:
        """Detect the encoding of a file."""
        with open(file_path, 'rb') as file:
            raw_data = file.read()
            result = chardet.detect(raw_data)
            return result['encoding']

    @staticmethod
    def read_csv_safely(file_path: str) -> pd.DataFrame:
        """Read CSV file with automatic encoding detection."""
        try:
            encoding = DataLoader.detect_encoding(file_path)
            return pd.read_csv(file_path, encoding=encoding)
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
            raise

    @staticmethod
    def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess dataframe."""
        df = df.replace(r'^\s*$', pd.NA, regex=True)
        
        # Handle missing values
        for column in df.columns:
            if df[column].dtype == 'object':
                df[column] = df[column].fillna('N/A')
            else:
                df[column] = df[column].fillna(0)
        
        # Clean text columns
        for column in df.select_dtypes(include=['object']).columns:
            df[column] = df[column].astype(str).apply(
                lambda x: x.encode('ascii', 'ignore').decode('ascii')
            )
        
        return df

    @staticmethod
    @st.cache_data
    def load_data(file_paths: List[str]) -> pd.DataFrame:
        """Load and combine multiple CSV files."""
        dataframes = []
        for file_path in file_paths:
            try:
                df = DataLoader.read_csv_safely(file_path)
                df = DataLoader.clean_dataframe(df)
                dataframes.append(df)
                st.sidebar.success(f"Loaded {file_path}")
            except Exception as e:
                st.sidebar.error(f"Failed to load {file_path}: {e}")
                logger.error(f"Error loading {file_path}: {e}")
                continue
        
        if not dataframes:
            raise ValueError("No data files were successfully loaded")
        
        return pd.concat(dataframes, ignore_index=True)

class ChatBot:
    """Handles chat functionality and response generation."""
    
    def __init__(self, model: EmbeddingModel, data: pd.DataFrame):
        self.model = model
        self.data = data
        self.context_embeddings = self._create_context_embeddings()

    def _create_context_embeddings(self) -> np.ndarray:
        """Create embeddings for the context data."""
        context_text = self.data.apply(
            lambda row: ' | '.join(map(str, row)), axis=1
        ).tolist()
        return self.model.encode(context_text)

    def find_relevant_context(self, query: str) -> pd.Series:
        """Find the most relevant context for a query."""
        query_embedding = self.model.encode([query])
        similarity_scores = util.pytorch_cos_sim(query_embedding, self.context_embeddings)
        most_similar_idx = int(similarity_scores.argmax())
        return self.data.iloc[most_similar_idx]

    def generate_response(self, query: str, conversation_history: List[Dict]) -> str:
        """Generate a response using the LLM."""
        try:
            relevant_context = self.find_relevant_context(query)
            context = self._format_context(conversation_history, relevant_context)
            
            response = ollama.chat(
                model='llama3',
                messages=[
                    {
                        "role": "system",
                        "content": "You are an AI assistant for ABAC university. "
                                 "Provide accurate and concise responses."
                    },
                    {
                        "role": "user",
                        "content": f"Context: {context}\n\nQuestion: {query}"
                    }
                ]
            )
            
            return response['message']['content']
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"I apologize, but I encountered an error: {str(e)}"

    def _format_context(self, history: List[Dict], relevant_context: pd.Series) -> str:
        """Format conversation history and context for the prompt."""
        history_text = "\n".join(
            f"{msg['role']}: {msg['content']}" for msg in history[-5:]
        )
        return f"{history_text}\nRelevant information: {relevant_context.to_string()}"

class UserInterface:
    """Handles Streamlit interface elements."""
    
    @staticmethod
    def display_messages(messages: List[Dict]):
        """Display chat messages."""
        st.markdown("### Chat History")
        for i, msg in enumerate(messages):
            UserInterface._display_message(msg, i)

    @staticmethod
    def _display_message(msg: Dict, index: int):
        """Display a single message with styling."""
        is_user = msg["role"] == "user"
        role_icon = "🧑" if is_user else "🤖"
        bg_color = "#011f4b" if is_user else "#1c1c1c"
        alignment = "right" if is_user else "left"
        
        st.markdown(
            f'<div style="background-color: {bg_color}; color: white; '
            f'padding: 10px; border-radius: 10px; margin-bottom: 10px; '
            f'max-width: 80%; float: {alignment};">'
            f'{role_icon}: {msg["content"]}</div>',
            unsafe_allow_html=True
        )
        
        if not is_user and msg["content"] != "How can I help you?":
            UserInterface._display_feedback_buttons(index)

    @staticmethod
    def _display_feedback_buttons(index: int):
        """Display feedback buttons for a message."""
        col1, col2 = st.columns(2)
        with col1:
            if st.button("👍", key=f"thumbs_up_{index}"):
                UserInterface._save_feedback(index, "Positive")
        with col2:
            if st.button("👎", key=f"thumbs_down_{index}"):
                feedback = st.text_input(
                    "How can we improve?",
                    key=f"feedback_{index}"
                )
                if st.button("Submit", key=f"submit_{index}"):
                    UserInterface._save_feedback(index, feedback)

    @staticmethod
    def _save_feedback(index: int, feedback: str):
        """Save user feedback to file."""
        try:
            with open('feedback.txt', 'a') as f:
                message = st.session_state.messages[index]['content']
                f.write(f"Message: {message}\nFeedback: {feedback}\n\n")
            st.success("Thank you for your feedback!")
        except Exception as e:
            logger.error(f"Error saving feedback: {e}")
            st.error("Failed to save feedback")

def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "How can I help you?"}
        ]
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []

def main():
    """Main application function."""
    st.title("ABAC University Assistant")
    
    # Initialize session state
    initialize_session_state()
    
    try:
        # Load data
        file_paths = ['CE.csv', 'Arjans.csv', 'combined_faqs.csv']
        data = DataLoader.load_data(file_paths)
        
        # Initialize models
        embedding_model = EmbeddingModel()
        chatbot = ChatBot(embedding_model, data)
        
        # Display chat interface
        UserInterface.display_messages(st.session_state.messages)
        
        # Input area
        st.markdown("### Ask a Question")
        col1, col2 = st.columns([4, 1])
        with col1:
            user_input = st.text_input("Your question:")
        with col2:
            if st.button("Send") and user_input:
                # Add user message
                st.session_state.messages.append(
                    {"role": "user", "content": user_input}
                )
                
                # Generate and add response
                response = chatbot.generate_response(
                    user_input,
                    st.session_state.conversation_history
                )
                st.session_state.messages.append(
                    {"role": "assistant", "content": response}
                )
                
                # Update conversation history
                st.session_state.conversation_history.append(
                    {"role": "user", "content": user_input}
                )
                st.session_state.conversation_history.append(
                    {"role": "assistant", "content": response}
                )
                
                st.rerun()
        
        # Common questions section
        st.markdown("### Common Questions")
        common_questions = [
            "What is the Map of ABAC?",
            "What are the free elective courses?",
            "What do I do if I miss Ethics Seminar?",
            "Do we have to register for English courses?"
        ]
        
        for question in common_questions:
            if st.button(question):
                st.session_state.messages.append(
                    {"role": "user", "content": question}
                )
                response = chatbot.generate_response(
                    question,
                    st.session_state.conversation_history
                )
                st.session_state.messages.append(
                    {"role": "assistant", "content": response}
                )
                st.rerun()
        
        # Clear chat button
        if st.sidebar.button("Clear Chat"):
            st.session_state.messages = [
                {"role": "assistant", "content": "How can I help you?"}
            ]
            st.session_state.conversation_history = []
            st.rerun()
            
    except Exception as e:
        logger.error(f"Application error: {e}")
        st.error("An error occurred. Please try again or contact support.")

if __name__ == "__main__":
    main()