import os
import streamlit as st
import tempfile
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings
from streamlit_option_menu import option_menu
from dotenv import load_dotenv, find_dotenv
from langsmith import Client
from py2neo import Graph
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
from src.helper import voice_input, llm_model_object, text_to_speech
# Import your custom Neo4j-based knowledge graph class
from KnowledgeGraph_Neo4j import RAG_Graph

# Load LangSmith API key and configure environment variables
load_dotenv(find_dotenv())
os.environ["LANGSMITH_API_KEY"] = str(os.getenv("LANGSMITH_API_KEY"))
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"

# Apply custom CSS styling for light theme and background image
def apply_light_theme_css():
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url('https://imgcdn.stablediffusionweb.com/2024/4/7/5f283b4a-5ef3-4a3a-8386-f3a1e812a324.jpg');
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            color:white;
            
        }

        .stChatMessage {
            border-radius: 10px;
            padding: 10px;
            margin: 5px 0;
        }

        .stChatMessage.user {
            background-color: #e3f2fd;
            text-align: left;
            color: black;
        }

        .stChatMessage.assistant {
            background-color: black;
            text-align: left;
            color: white;  /* Change text color to white */
        }

        div[data-baseweb="input"] > div {
            border-radius: 10px;
            background-color: #ffffff;
            color: black;
            border: 1px solid #b0bec5;
        }

        input::placeholder {
            color: #b0bec5;
        }
        

        h1 {
            color: white;
            text-align: center;
        }

        ::-webkit-scrollbar {
            width: 8px;
        }
        ::-webkit-scrollbar-thumb {
            background-color: #b0bec5;
            border-radius: 10px;
        }

        /* Informational section styling */
        .info-section {
            background-color: rgba(white);
            padding: 15px;
            border-radius: 10px;
            margin-top: 20px;
            box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.8);
        }

        .info-section h3 {
            color: orange;
        }
        .info-section p {
            color: white;
        }

        </style>
        """,
        unsafe_allow_html=True
    )


# Chatbot function
def RAG_Neo4j():
    # Neo4j-based chatbot function
    
    rag_graph = RAG_Graph()
    if "messages1" not in st.session_state:
        st.session_state.messages1 = []

    for message in st.session_state.messages1:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if st.button("🎙️"):
        with st.spinner("Listening..."):
            text = voice_input()
            response = llm_model_object(text)
            text_to_speech(response)
            
            audio_file = open("speech.mp3", "rb")
            audio_bytes = audio_file.read()
            
            st.text_area(label="Response:", value=response, height=350)
            st.audio(audio_bytes)
            st.download_button(label="Download Speech",
                               data=audio_bytes,
                               file_name="speech.mp3",
                               mime="audio/mp3")

    if prompt1 := st.chat_input("Ask a question to the document assistant"):
        st.chat_message("user").markdown(prompt1)
        st.session_state.messages1.append({"role": "user", "content": prompt1})

        response1 = rag_graph.ask_question_chain(prompt1)

        with st.chat_message("assistant"):
            st.markdown(response1)

        st.session_state.messages1.append({"role": "assistant", "content": response1})

# Home page function with image and chatbot description
def home():
   
        
    st.markdown(
        """
        <div class="info-section">
            <h3>Chhatrapati Shivaji Maharaj AI Chatbot</h3>
            <p>This chatbot is dedicated to providing information about the life, achievements, and legacy of Chhatrapati Shivaji Maharaj. 
        Feel free to ask any questions related to his history, military strategies, governance, and more.</p>
            <p>Use the Chat option in the menu to start interacting with the chatbot.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Chat page function with image and chatbot
def chat_page():
    image_url = "https://img.freepik.com/premium-vector/shivaji-maharaj-shaniwar-wada-fort-maharashtra-vector_1076263-676.jpg"
    apply_light_theme_css()
    
    left_column, right_column = st.columns([1, 2])

    with left_column:
        st.image(image_url, caption="Chhatrapati Shivaji Maharaj", use_container_width=True)

    with right_column:
        RAG_Neo4j()

# Main function for the Streamlit app
def main():
    apply_light_theme_css()

    st.title("Chat With Me On Chhatrapati Shivaji Maharaj")

    option = option_menu(
        menu_title="Main Menu",  # Main title of the menu bar
        options=["Home", "Chat"],  # List of options
        icons=["house", "chat"],  # Icons for each option
        menu_icon="cast",  # Icon for the main menu
        default_index=0,  # Set the default selected option (first option in this case)
        orientation="horizontal",  # You can change this to "vertical" if you prefer a vertical layout
        styles={
            "container": {
                "padding": "15px",
                "background-color" : "rgba(white)",  # Adds space around the menu
                "margin-top" : "0px",  # Light background color for the menu container
                "border-radius": "5px",
                "box-shadow": "0px 0px 10px rgba(0, 0, 0, 0.8)",  # Rounded corners for the menu
            },
            "icon": {
                "font-size": "18px",  # Icon size
                "color": "gray",  # Icon color
            },
            "nav-link": {
                "font-size": "16px",  # Font size for the menu text
                "text-align": "center",  # Aligns text in the center
                "color": "orange",  # Text color
                "padding": "10px",  # Adds padding to each menu item
            },
            "nav-link-selected": {
                "background-color": "grey",  # Green background for the selected item
                "color": "white",  # White text for the selected item
                "border-radius": "5px",  # Rounded corners for the selected item
            },
        }
    )

    if option == "Home":
        home()
    elif option == "Chat":
        chat_page()

if __name__ == "__main__":
    main()
