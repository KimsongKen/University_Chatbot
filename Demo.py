import os
import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import ollama

# Set the PandasAI API key
os.environ['PANDASAI_API_KEY'] = '$2a$10$LB0ULOGAqZGxAqEJ90HOeuNoKKI1x0NyrfKKplU7Klk3WJcwmsZZK'

st.title("💬 Local LLMBot")

# Initialize the session state for storing messages and other necessary components
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]
    st.session_state["faq_data"] = None  # Initialize to store FAQ DataFrame
    st.session_state["faq_embeddings"] = None  # Store FAQ embeddings
    st.session_state["model"] = None  # Store the trained model
    st.session_state["other_data"] = []  # Store other data without FAQs

# Display chat messages
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.chat_message(msg["role"], avatar="🧑‍💻").write(msg["content"])
    else:
        st.chat_message(msg["role"], avatar="🤖").write(msg["content"])

# Function to load datasets from multiple file paths and handle FAQ and non-FAQ data
def load_and_encode_faqs(file_paths):
    faq_dataframes = []
    other_dataframes = []

    for file_path in file_paths:
        try:
            df = pd.read_csv(file_path)
            if 'Question' in df.columns and 'Answer' in df.columns:
                faq_dataframes.append(df)
                st.write(f"FAQs loaded successfully from: {file_path}")
            else:
                other_dataframes.append(df)
                st.write(f"File {file_path} does not have the required FAQ columns. Loaded as other data.")
        except Exception as e:
            st.write(f"Error loading {file_path}: {e}")

    # Combine and encode FAQs if available
    if faq_dataframes:
        combined_faq_df = pd.concat(faq_dataframes, ignore_index=True)
        model = SentenceTransformer('all-MiniLM-L6-v2')
        faq_embeddings = model.encode(list(combined_faq_df['Question']))
        st.session_state["faq_data"] = combined_faq_df
        st.session_state["faq_embeddings"] = faq_embeddings
        st.session_state["model"] = model
        st.write("Combined FAQ Data:")
        st.dataframe(combined_faq_df)

    # Store other data if available
    if other_dataframes:
        st.session_state["other_data"] = other_dataframes
        st.write("Other Data Loaded:")
        for i, df in enumerate(other_dataframes):
            st.write(f"Dataset {i+1}:")
            st.dataframe(df)  # Display each DataFrame for review
            st.write("Columns:", df.columns.tolist())  # Show the column names for understanding

# Load the CSV files during app initialization
file_paths = ['combined_faqs.csv', 'Arjans.csv', 'CE.csv']
load_and_encode_faqs(file_paths)

# Function to search FAQs
def search_faqs(query):
    model = st.session_state["model"]
    faq_embeddings = st.session_state["faq_embeddings"]
    data = st.session_state["faq_data"]
    
    if model is None or faq_embeddings is None or data is None:
        return "I don't have any FAQs loaded to answer your query."
    
    user_question_embedding = model.encode([query])
    similarity_scores = util.pytorch_cos_sim(user_question_embedding, faq_embeddings)
    
    # Convert the most similar index to an integer
    most_similar_idx = int(similarity_scores.argmax())
    
    most_similar_question = data.iloc[most_similar_idx]['Question']
    most_similar_answer = data.iloc[most_similar_idx]['Answer']
    most_similar_image = data.iloc[most_similar_idx]['Image'] if 'Image' in data.columns else None
    
    return most_similar_question, most_similar_answer, most_similar_image

# Function to explore and interact with other datasets
def explore_data():
    st.subheader("Data Exploration and Insights")
    for i, df in enumerate(st.session_state["other_data"]):
        st.write(f"Exploring Dataset {i+1}")
        
        # Show basic information
        if st.button(f"Show Summary of Dataset {i+1}"):
            st.write(df.describe(include='all'))  # Show summary stats of the DataFrame
            st.write("Data Types:", df.dtypes)  # Show data types of columns

        # Column Exploration
        selected_column = st.selectbox(f"Select a column from Dataset {i+1}:", df.columns)
        if selected_column:
            st.write(f"Exploring column: {selected_column}")
            st.write("Unique Values:", df[selected_column].unique())
            if df[selected_column].dtype == 'object':
                st.write("Value Counts:", df[selected_column].value_counts())

            # Filtering options
            filter_value = st.text_input(f"Filter {selected_column} by value in Dataset {i+1}:")
            if filter_value:
                filtered_data = df[df[selected_column].astype(str).str.contains(filter_value, case=False, na=False)]
                st.write(f"Filtered Data based on {filter_value}:")
                st.dataframe(filtered_data)

        # Generate Insights or Recommendations
        if st.button(f"Generate Insights for Dataset {i+1}"):
            insights = generate_insights(df)
            st.write(insights)

# Function to generate insights from a DataFrame (basic example)
def generate_insights(df):
    # Example insights: Top categories, mean values of numerical columns, etc.
    insights = []
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            insights.append(f"The average value of {col} is {df[col].mean():.2f}.")
        elif df[col].dtype == 'object':
            top_value = df[col].value_counts().idxmax()
            insights.append(f"The most common value in {col} is '{top_value}'.")
    return "\n".join(insights)

# Function to generate responses from user queries
def generate_response(prompt):
    # Check if FAQs are loaded and search within them
    if st.session_state["faq_data"] is not None:
        question, answer, image_url = search_faqs(prompt)
        
        # Display the answer
        st.chat_message("assistant", avatar="🤖").write(f"Q: {question}\nA: {answer}")
        
        # If there is an image URL, display the image
        if image_url and pd.notna(image_url):
            st.image(image_url, caption="Relevant Map")
        
        # Store the Q&A in session state for history
        st.session_state.messages.append({"role": "assistant", "content": f"Q: {question}\nA: {answer}"})
        
        return answer
    else:
        # If no FAQ data is loaded, respond using Ollama model
        response = ollama.chat(model='llama3', stream=True, messages=st.session_state.messages)
        full_message = ""
        for partial_resp in response:
            token = partial_resp["message"]["content"]
            full_message += token
        
        # Store the response in session state for history
        st.session_state.messages.append({"role": "assistant", "content": full_message})
        
        return full_message

# Handling user input for chatbot interaction
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user", avatar="🧑‍💻").write(prompt)
    
    response = generate_response(prompt)
    
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.chat_message("assistant", avatar="🤖").write(response)

# Call the explore_data function to enable data exploration in the UI
explore_data()
