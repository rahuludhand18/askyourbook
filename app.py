"""
Project: AskYourBook 📖
Description: An intelligent RAG (Retrieval Augmented Generation) system for interacting with PDF documents.
Author: Rahul R Udhand
Version: 3.0.1 (Stable - Fixed Imports)
"""

# ==============================================================================
# 1. IMPORTS & SETUP
# ==============================================================================
import os
import time
import tempfile
import logging
from typing import List, Tuple, Optional, Generator

# UI & Environment
import streamlit as st
from dotenv import load_dotenv

# Machine Learning & NLP
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
# FIXED: Added this missing import
from langchain_core.embeddings import Embeddings

# API Integration
from groq import Groq

# Set up simple logging to track events in the console (helps with debugging)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables (API keys)
load_dotenv()

# ==============================================================================
# 2. APP CONFIGURATION & CONSTANTS
# ==============================================================================

st.set_page_config(
    page_title="AskYourBook 📖",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration settings for text splitting
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# Available Models
MODEL_OPTIONS = {
    "Llama 3.3 70B (High Intelligence)": "llama-3.3-70b-versatile",
    "Llama 3.1 8B (High Speed)": "llama-3.1-8b-instant",
}

# Supported Languages for Output
LANGUAGE_OPTIONS = {
    "English": "English",
    "Hindi (हिंदी)": "Hindi (Devanagari Script)",
    "Kannada (ಕನ್ನಡ)": "Kannada",
    "Spanish (Español)": "Spanish",
    "French (Français)": "French"
}

# ==============================================================================
# 3. UI STYLING (The "Lovable" Theme)
# ==============================================================================

def inject_custom_css():
    """
    Injects custom CSS to override Streamlit's default look.
    Targeting a 'Lovable', soft-dark aesthetic with gradients and rounded corners.
    """
    st.markdown("""
    <style>
        /* --- Fonts & Global Colors --- */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
        
        :root {
            --bg-color: #121212;
            --card-bg: #1E1E1E;
            --accent-gradient: linear-gradient(135deg, #FF6B6B 0%, #556270 100%);
            --button-gradient: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            --text-primary: #E0E0E0;
            --text-secondary: #B0B0B0;
        }

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
            color: var(--text-primary);
        }

        /* --- Main Background --- */
        .stApp {
            background-color: var(--bg-color);
        }

        /* --- Sidebar Styling --- */
        [data-testid="stSidebar"] {
            background-color: #0d0d0d;
            border-right: 1px solid #2d2d2d;
        }

        /* --- Custom Headers --- */
        h1, h2, h3 {
            font-weight: 800;
            background: -webkit-linear-gradient(45deg, #A0C4FF, #CDB4DB);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        /* --- Cards (Glassmorphismish) --- */
        .stCard, div[data-testid="stExpander"] {
            background-color: var(--card-bg);
            border-radius: 16px;
            border: 1px solid #333;
            padding: 20px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            transition: all 0.3s ease;
        }
        
        div[data-testid="stExpander"]:hover {
            border-color: #667eea;
        }

        /* --- Buttons --- */
        .stButton > button {
            background: var(--button-gradient);
            color: white;
            border: none;
            border-radius: 12px;
            padding: 0.5rem 1rem;
            font-weight: 600;
            letter-spacing: 0.5px;
            transition: transform 0.2s;
        }

        .stButton > button:hover {
            transform: scale(1.02);
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        }

        /* --- Chat Messages --- */
        .user-message {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 12px 18px;
            border-radius: 18px 18px 4px 18px;
            margin-bottom: 10px;
            width: fit-content;
            margin-left: auto;
            max-width: 80%;
            box-shadow: 0 2px 10px rgba(118, 75, 162, 0.3);
        }

        .bot-message {
            background-color: #2D2D2D;
            color: #E0E0E0;
            padding: 12px 18px;
            border-radius: 18px 18px 18px 4px;
            margin-bottom: 10px;
            width: fit-content;
            max-width: 80%;
            border: 1px solid #444;
        }

        /* --- Input Field --- */
        .stTextInput input {
            background-color: #1E1E1E;
            border-radius: 12px;
            border: 1px solid #333;
            color: white;
            padding: 10px;
        }
        .stTextInput input:focus {
            border-color: #764ba2;
        }

        /* --- Spinner --- */
        .stSpinner > div {
            border-top-color: #764ba2 !important;
        }
    </style>
    """, unsafe_allow_html=True)

# ==============================================================================
# 4. BACKEND LOGIC & CLASSES
# ==============================================================================

class LocalEmbeddingManager(Embeddings):
    """
    Manages local embeddings using Sentence Transformers.
    We implement the Embeddings interface from LangChain to ensure compatibility.
    """
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        try:
            self.model = SentenceTransformer(model_name)
            logger.info(f"Embedding model '{model_name}' loaded successfully.")
        except Exception as e:
            st.error(f"Failed to load embedding model. Please check internet connection. Error: {e}")
            logger.error(f"Embedding load error: {e}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Generate embeddings for a list of documents
        embeddings = self.model.encode(texts)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        # Generate embedding for a single user query
        embedding = self.model.encode([text])[0]
        return embedding.tolist()


class GroqInferenceEngine:
    """
    Wrapper for the Groq API.
    Handles the communication with the Large Language Model (LLM).
    """
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        
        # Security Check
        if not api_key:
            st.error("⚠️ API Key Missing! Please add GROQ_API_KEY to your .env file.")
            st.stop()
            
        self.client = Groq(api_key=api_key)

    def generate_response(self, 
                          system_prompt: str, 
                          user_prompt: str, 
                          model_id: str, 
                          temperature: float = 0.3) -> str:
        """
        Sends a request to Groq and returns the response string.
        Includes error handling for rate limits or API downtime.
        """
        try:
            completion = self.client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                max_tokens=4096, # Allowing for long responses
                top_p=1,
                stream=False
            )
            return completion.choices[0].message.content
            
        except Exception as e:
            error_msg = f"Inference Error: {str(e)}"
            logger.error(error_msg)
            return f"😓 Oops! Something went wrong with the AI engine. \n\nError details: {str(e)}"


class DataProcessor:
    """
    Handles the messy work of file I/O and processing.
    1. Saves uploaded file temporarily.
    2. Reads PDF.
    3. Splits text into chunks.
    4. Indexes into Vector Database.
    """
    def __init__(self):
        self.embedding_manager = LocalEmbeddingManager()

    def ingest_pdf(self, uploaded_file) -> Tuple[Chroma, str]:
        """
        Main pipeline for ingesting a PDF.
        Returns: (VectorStore Object, Full Text String)
        """
        # Save uploaded file to a temporary location so PyPDFLoader can read it
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        try:
            # Step 1: Load
            loader = PyPDFLoader(tmp_path)
            pages = loader.load()
            
            # Combine all text for the 'Summary' and 'Podcast' features
            full_corpus = "\n\n".join([page.page_content for page in pages])

            # Step 2: Split
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                separators=["\n\n", "\n", ".", " ", ""]
            )
            chunks = splitter.split_documents(pages)
            
            logger.info(f"Document split into {len(chunks)} chunks.")

            # Step 3: Index
            # Create a unique collection name to prevent session collision
            collection_id = f"user_repo_{int(time.time())}"
            
            vector_db = Chroma.from_documents(
                documents=chunks,
                embedding=self.embedding_manager,
                collection_name=collection_id
            )

            return vector_db, full_corpus

        except Exception as e:
            logger.error(f"Processing failed: {e}")
            raise e
        finally:
            # Cleanup: Remove temp file to keep the server clean
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

# ==============================================================================
# 5. CORE LOGIC (RAG & PROMPTING)
# ==============================================================================

class AskYourBookAgent:
    """
    The main controller class that connects UI inputs to Backend logic.
    """
    def __init__(self, vector_db, full_text):
        self.vector_db = vector_db
        self.full_text = full_text
        self.llm_engine = GroqInferenceEngine()
        self.retriever = self.vector_db.as_retriever(search_kwargs={"k": 5})

    def query_document(self, query: str, model_id: str, language: str) -> Tuple[str, List[Document]]:
        """
        Performs the RAG (Retrieval Augmented Generation) loop.
        """
        # 1. Retrieval
        retrieved_docs = self.retriever.invoke(query)
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])

        # 2. System Prompt construction
        system_prompt = f"""
        You are AskYourBook AI, a friendly and extremely intelligent research assistant.
        
        Guidelines:
        - Answer the user's question using ONLY the provided Context.
        - If the answer is not in the context, politely say you don't know.
        - Your answer must be in the following language: {language}.
        - Use emojis to make the answer friendly if appropriate.
        - Structure your answer with clear headings or bullet points.
        """

        user_prompt = f"""
        Context from Document:
        {context}

        Question:
        {query}
        """

        # 3. Generation
        answer = self.llm_engine.generate_response(system_prompt, user_prompt, model_id)
        return answer, retrieved_docs

    def generate_study_guide(self, model_id: str, language: str) -> str:
        """
        Generates a summary/study guide based on the full text (truncated to fit context).
        """
        # Safety truncate to avoid token limits (approx 30k chars is safe for Llama 3 70B)
        safe_text = self.full_text[:32000]

        system_prompt = "You are an expert academic tutor."
        user_prompt = f"""
        Analyze the following text and generate a comprehensive Study Guide in {language}.
        
        Please include:
        1. 📝 **Executive Summary** (A concise overview)
        2. 🧠 **Key Concepts** (Bullet points of main ideas)
        3. 📖 **Vocabulary Builder** (Difficult terms explained)
        4. ❓ **Quiz Yourself** (3 Short answer questions)

        Text to analyze:
        {safe_text}
        """
        
        return self.llm_engine.generate_response(system_prompt, user_prompt, model_id)

    def generate_podcast_script(self, model_id: str, language: str) -> str:
        """
        Generates a conversational script.
        """
        safe_text = self.full_text[:32000]

        system_prompt = "You are a creative scriptwriter for a popular tech/science podcast."
        user_prompt = f"""
        Create a podcast script between two hosts, 'Alex' (Curious, energetic) and 'Jamie' (Expert, calm).
        They are discussing the document below.
        
        - Keep it conversational and fun.
        - Use sound effects cues like [SFX: Intro Music].
        - Language: {language}
        
        Document content:
        {safe_text}
        """

        return self.llm_engine.generate_response(system_prompt, user_prompt, model_id, temperature=0.7)

# ==============================================================================
# 6. FRONTEND (STREAMLIT UI)
# ==============================================================================

def init_session():
    """Initializes session state variables if they don't exist."""
    if "agent" not in st.session_state:
        st.session_state.agent = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "processing_complete" not in st.session_state:
        st.session_state.processing_complete = False

def render_sidebar():
    """Renders the sidebar controls."""
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        
        # 1. Model Selection
        st.markdown("### 🧠 AI Model")
        selected_model_label = st.selectbox(
            "Choose your brain:", 
            list(MODEL_OPTIONS.keys()), 
            index=0,
            label_visibility="collapsed"
        )
        
        # 2. Language Selection
        st.markdown("### 🗣️ Language")
        selected_lang_label = st.selectbox(
            "Output Language:",
            list(LANGUAGE_OPTIONS.keys()),
            index=0,
            label_visibility="collapsed"
        )

        st.markdown("---")
        
        # 3. File Upload
        st.markdown("### 📂 Upload Book/PDF")
        uploaded_file = st.file_uploader(
            "Drag & drop PDF here", 
            type=["pdf"], 
            label_visibility="collapsed"
        )

        # 4. Process Button logic
        if uploaded_file and not st.session_state.processing_complete:
            if st.button("🚀 Analyze Document", use_container_width=True):
                with st.spinner("Reading... Splitting... Indexing..."):
                    try:
                        processor = DataProcessor()
                        vector_db, full_text = processor.ingest_pdf(uploaded_file)
                        
                        # Initialize the Agent
                        st.session_state.agent = AskYourBookAgent(vector_db, full_text)
                        st.session_state.processing_complete = True
                        
                        st.success("Analysis Complete!")
                        time.sleep(1) # Visual feedback pause
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to process file: {e}")

        # Reset Button
        if st.session_state.processing_complete:
            if st.button("🔄 Start Over", type="secondary", use_container_width=True):
                st.session_state.clear()
                st.rerun()

        # Return selected configuration
        return MODEL_OPTIONS[selected_model_label], LANGUAGE_OPTIONS[selected_lang_label]

def render_chat_tab(model_id, language):
    """Handles the main Chat interface."""
    st.markdown("### 💬 Chat with your Book")
    
    # Display History
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"<div class='user-message'>{msg['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='bot-message'>{msg['content']}</div>", unsafe_allow_html=True)

    # Chat Input
    if prompt := st.chat_input("Ask something specific about the document..."):
        # Add User Message
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.markdown(f"<div class='user-message'>{prompt}</div>", unsafe_allow_html=True)

        # Generate Reply
        with st.spinner("Thinking..."):
            agent = st.session_state.agent
            response, docs = agent.query_document(prompt, model_id, language)
            
            # Add Bot Message
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.markdown(f"<div class='bot-message'>{response}</div>", unsafe_allow_html=True)

            # Debug Context (Optional, kept in expander for cleanness)
            with st.expander("🔍 View Sources (Debug Info)"):
                for i, doc in enumerate(docs):
                    st.markdown(f"**Source {i+1}:**")
                    st.caption(doc.page_content[:300] + "...")

def render_study_tab(model_id, language):
    """Handles the Study Guide generation interface."""
    st.markdown("### 📝 Smart Study Guide")
    st.caption("Generate a structured summary, vocabulary list, and quiz from your document.")
    
    col1, col2 = st.columns([1, 4])
    with col1:
        # Just a visual spacer or icon if needed
        pass 
    
    if st.button("✨ Create Study Guide Now", use_container_width=True):
        with st.spinner("Analyzing document structure... this may take a moment"):
            agent = st.session_state.agent
            guide = agent.generate_study_guide(model_id, language)
            st.markdown(guide)

def render_podcast_tab(model_id, language):
    """Handles the Podcast Script interface."""
    st.markdown("### 🎙️ AI Podcast Studio")
    st.caption("Turn your dry document into an engaging conversation script.")

    if st.button("🎧 Generate Script", use_container_width=True):
        with st.spinner("Writing script for Alex and Jamie..."):
            agent = st.session_state.agent
            script = agent.generate_podcast_script(model_id, language)
            
            # Display nicely
            st.markdown("---")
            st.markdown(script)
            
            # Download option
            st.download_button(
                label="📥 Download Script (.txt)",
                data=script,
                file_name="podcast_script.txt",
                mime="text/plain"
            )

# ==============================================================================
# 7. MAIN ENTRY POINT
# ==============================================================================

def main():
    # 1. Load CSS
    inject_custom_css()
    
    # 2. Init Session
    init_session()
    
    # 3. Render Sidebar & Get Config
    model_id, language = render_sidebar()

    # 4. Main Content Area
    # If no file is processed yet, show the Welcome Landing Page
    if not st.session_state.processing_complete:
        st.markdown("""
        <div style="text-align: center; padding-top: 50px; padding-bottom: 50px;">
            <h1 style="font-size: 3.5rem;">AskYourBook 📖</h1>
            <p style="font-size: 1.2rem; color: #B0B0B0;">
                Transform your PDFs into interactive conversations, study guides, and podcasts.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Display Feature Cards (Only visual info, no functionality logic here)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div class="stCard">
                <h3>🗣️ Chat</h3>
                <p>Ask questions and get instant answers sourced directly from your PDF.</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="stCard">
                <h3>📚 Study</h3>
                <p>Auto-generate vocabulary lists, key concepts, and quizzes.</p>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown("""
            <div class="stCard">
                <h3>🎙️ Listen</h3>
                <p>Convert boring text into fun, conversational podcast scripts.</p>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<br><p style='text-align:center'>👈 <em>Start by uploading a file in the sidebar!</em></p>", unsafe_allow_html=True)

    else:
        # File is processed, show the main tools
        tab1, tab2, tab3 = st.tabs(["💬 Chat", "📝 Study Guide", "🎙️ Podcast"])
        
        with tab1:
            render_chat_tab(model_id, language)
        with tab2:
            render_study_tab(model_id, language)
        with tab3:
            render_podcast_tab(model_id, language)

if __name__ == "__main__":
    main()