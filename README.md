AskYourBook is an advanced AI-powered document assistant designed to transform the way we interact with static PDFs. 
It uses Retrieval-Augmented Generation (RAG) to convert any uploaded document into an interactive, searchable knowledge base.
The application goes beyond simple question-answering by automatically generating study guides, vocabulary lists, and conversational podcast scripts, helping users learn faster and retain more information.

Key Features

Interactive RAG Chat: Users can chat naturally with their document while the system retrieves accurate, context-specific responses with citations.
Smart Study Guide: Automatically produces an executive summary, key concepts, vocabulary builder, and quiz questions from the document.
AI Podcast Agent: Converts dense or technical content into a friendly dialogue script between two virtual hosts, suitable for audio-style learning.
Multi-Language Output: Users can ask questions in English and receive responses in Hindi, Kannada, French, or Spanish.
Ultra-Low Latency: Powered by Groq’s inference engine running Llama 3.3 70B for fast, responsive answers.
Privacy-First Design: Embedding models and vector storage run locally using SentenceTransformers and ChromaDB.
Modern UI: Includes a custom glassmorphic dark-theme interface for an elegant user experience.

Tech Stack
LLM Engine: Llama 3.3 70B via Groq API
Embeddings: sentence-transformers/all-MiniLM-L6-v2
Vector Database: ChromaDB (Local storage)
Orchestration: LangChain Core and Community

Frontend: Streamlit

PDF Processing: PyPDFLoader
