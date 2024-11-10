import os
import pandas as pd
import streamlit as st
import requests
from dotenv import load_dotenv

# Load Groq Cloud API Key
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# Set the correct Groq Cloud API endpoint
groq_endpoint = "https://api.groq.com/openai/v1/chat/completions"  # Replace with actual endpoint if needed

def load_data(uploaded_file):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Dataset Overview:")
        st.write(df.head())
        return df
    else:
        st.write("Please upload a dataset.")
        return None

def analyze_data(df):
    summary = {
        "columns": df.columns.tolist(),
        "data_types": df.dtypes.to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "num_rows": len(df),
        "num_columns": len(df.columns)
    }
    st.write("Data Summary:", summary)
    return summary

def generate_ml_code(prompt):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # Update to use 'messages' instead of 'prompt' for chat models
    data = {
        "model": "gemma2-9b-it",  # Replace with your desired model ID
        "messages": [
            {"role": "system", "content": "You are an AI model that generates machine learning code."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 500,  # Adjust according to the model's capacity
        "temperature": 0.7,
        "top_p": 1,
        "stop": "\n"  # Optional stop sequence
    }
    
    response = requests.post(groq_endpoint, headers=headers, json=data)
    
    if response.status_code == 200:
        return response.json().get("choices", [])[0].get("message", {}).get("content", "")
    else:
        # Enhanced error message for better feedback
        st.error(f"Error {response.status_code}: {response.json().get('error', 'Unknown error')}")
        return None

def main():
    st.title("Automated ML Code Generator with Groq Cloud")
    
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        
        if df is not None:
            summary = analyze_data(df)
            
            task_type = st.selectbox("Select ML Task", ["Classification", "Regression"])
            
            if st.button("Generate Code"):
                prompt = (f"Generate Python code for {task_type} on a dataset with the following structure: "
                          f"{summary}. Include data preprocessing, model training, and evaluation.")
                
                code = generate_ml_code(prompt)
                if code:
                    st.code(code, language="python")

if __name__ == "__main__":
    main()