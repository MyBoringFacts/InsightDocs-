import streamlit as st
import pdfplumber
import docx
import os
import re
import numpy as np
import google.generativeai as palm
import logging
import time
import uuid
import json
import firebase_admin
from firebase_admin import credentials, firestore
from sklearn.metrics.pairwise import cosine_similarity

# Initialize Firebase
def init_firebase():
    if not firebase_admin._apps:
        data = json.loads(os.getenv("FIREBASE_CRED"))
        cred = credentials.Certificate(data)
        firebase_admin.initialize_app(cred)

init_firebase()
fs_client = firestore.client()

def save_conversation_to_firestore(session_id, user_question, assistant_answer, feedback=None):
    conv_ref = fs_client.collection("sessions").document(session_id).collection("conversations")
    data = {
        "user_question": user_question,
        "assistant_answer": assistant_answer,
        "feedback": feedback,
        "timestamp": firestore.SERVER_TIMESTAMP
    }
    doc_ref = conv_ref.add(data)
    return doc_ref[1].id

def save_message_to_firestore(session_id, role, content, feedback=None):
    messages_ref = fs_client.collection("sessions").document(session_id).collection("messages")
    data = {
        "role": role,
        "content": content,
        "feedback": feedback,
        "timestamp": firestore.SERVER_TIMESTAMP
    }
    doc_ref = messages_ref.add(data)
    return doc_ref[1].id

def handle_feedback(feedback_val):
    update_feedback_in_firestore(
        st.session_state.session_id,
        st.session_state.latest_conversation_id,
        feedback_val
    )
    st.session_state.conversations[-1]["feedback"] = feedback_val

def fetch_messages_from_firestore(session_id):
    messages_ref = fs_client.collection("sessions").document(session_id).collection("messages")
    docs = messages_ref.order_by("timestamp").stream()
    messages = []
    for doc in docs:
        data = doc.to_dict()
        data["id"] = doc.id
        messages.append(data)
    return messages

def update_feedback_in_firestore(session_id, conversation_id, feedback):
    conv_doc = fs_client.collection("sessions").document(session_id).collection("conversations").document(conversation_id)
    conv_doc.update({"feedback": feedback})

class Config:
    CHUNK_WORDS = 300
    EMBEDDING_MODEL = "models/gemini-embedding-exp-03-07"
    TOP_N = 5
    SYSTEM_PROMPT = (
        "You are a helpful assistant. Answer the question using the provided context below. "
        "Answer based on your knowledge if the context given is not enough."
    )
    GENERATION_MODEL = "models/gemini-1.5-flash"

API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    st.error("Google API key is not configured.")
    st.stop()
palm.configure(api_key=API_KEY)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@st.cache_data(show_spinner=True)
def generate_embedding_cached(text: str) -> list:
    logger.info("Calling API for embedding generation. Text snippet: %s", text[:50])
    try:
        response = palm.embed_content(
            model=Config.EMBEDDING_MODEL,
            content=text,
            task_type="retrieval_document"
        )
        if "embedding" not in response or not response["embedding"]:
            logger.error("No embedding returned from API.")
            st.error("No embedding returned. Please verify your API settings and input text.")
            return [0.0] * 768
        embedding = np.array(response["embedding"])
        if embedding.ndim == 2:
            embedding = embedding.flatten()
        elif embedding.ndim > 2:
            logger.error("Embedding has more than 2 dimensions.")
            st.error("Invalid embedding dimensions. Please check the API response.")
            return [0.0] * 768
        return embedding.tolist()
    except Exception as e:
        logger.error("Embedding generation failed: %s", e)
        st.error(f"Embedding generation failed: {e}")
        return [0.0] * 768

def generate_embedding(text: str) -> np.ndarray:
    embedding_list = generate_embedding_cached(text)
    return np.array(embedding_list)

def extract_text_from_file(uploaded_file) -> str:
    file_name = uploaded_file.name.lower()
    if file_name.endswith(".txt"):
        logger.info("Processing TXT file.")
        return uploaded_file.read().decode("utf-8")
    elif file_name.endswith(".pdf"):
        logger.info("Processing PDF file.")
        with pdfplumber.open(uploaded_file) as pdf:
            text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
            if not text:
                logger.error("PDF extraction returned empty text.")
            return text
    elif file_name.endswith(".docx"):
        logger.info("Processing DOCX file.")
        doc = docx.Document(uploaded_file)
        text = "\n".join([para.text for para in doc.paragraphs])
        if not text:
            logger.error("DOCX extraction returned empty text.")
        return text
    else:
        raise ValueError("Unsupported file type. Please upload a .txt, .pdf, or .docx file.")

def chunk_text(text: str) -> list[str]:
    max_words = Config.CHUNK_WORDS
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current_chunk = ""
    current_word_count = 0
    for paragraph in paragraphs:
        para_word_count = len(paragraph.split())
        if para_word_count > max_words:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
                current_word_count = 0
            sentences = re.split(r'(?<=[.!?])\s+', paragraph)
            temp_chunk = ""
            temp_word_count = 0
            for sentence in sentences:
                sentence_word_count = len(sentence.split())
                if temp_word_count + sentence_word_count > max_words:
                    if temp_chunk:
                        chunks.append(temp_chunk.strip())
                    temp_chunk = sentence + " "
                    temp_word_count = sentence_word_count
                else:
                    temp_chunk += sentence + " "
                    temp_word_count += sentence_word_count
            if temp_chunk:
                chunks.append(temp_chunk.strip())
        else:
            if current_word_count + para_word_count > max_words:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = paragraph + "\n\n"
                current_word_count = para_word_count
            else:
                current_chunk += paragraph + "\n\n"
                current_word_count += para_word_count
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def process_document(uploaded_file) -> None:
    try:
        file_text = extract_text_from_file(uploaded_file)
        if not file_text.strip():
            logger.error("Uploaded file contains no valid text.")
            st.error("The uploaded file contains no valid text.")
            return
        chunks = chunk_text(file_text)
        if not chunks:
            logger.error("No chunks generated from text.")
            st.error("Failed to split text into chunks.")
            return
        embeddings = [generate_embedding(chunk) for chunk in chunks]
        if all(np.all(embedding == 0) for embedding in embeddings):
            logger.error("All embeddings are zero vectors.")
            st.error("Failed to generate valid embeddings.")
            return
        doc_entry = {
            "file_name": uploaded_file.name,
            "document_text": file_text,
            "document_chunks": chunks,
            "document_embeddings": embeddings,
        }
        if "documents" not in st.session_state:
            st.session_state["documents"] = []
        st.session_state.documents.append(doc_entry)
        st.session_state.doc_processed = True
        st.success(f"Document '{uploaded_file.name}' processing complete! You can now start chatting.")
    except Exception as e:
        logger.error("Document processing failed: %s", e)
        st.error(f"An error occurred while processing the document: {e}")

def clear_documents():
    # Clear attached documents and chat messages from session state.
    if "documents" in st.session_state:
        del st.session_state["documents"]
    if "conversations" in st.session_state:
        del st.session_state["conversations"]
    # Update the dynamic key for the file uploader to force reinitialization.
    st.session_state["uploaded_files_key"] = str(uuid.uuid4())
    st.session_state.doc_processed = False
    st.success("All documents and chat messages have been cleared.")

def search_query(query: str) -> list[tuple[str, float]]:
    if "documents" not in st.session_state or len(st.session_state["documents"]) == 0:
        logger.error("No valid document embeddings found in session state.")
        st.error("No valid document embeddings found. Please upload a valid document.")
        return []
    query_embedding = generate_embedding(query)
    if np.all(query_embedding == 0):
        logger.error("Query embedding is a zero vector.")
        st.error("Failed to generate a valid query embedding.")
        return []
    query_embedding = query_embedding.reshape(1, -1)
    all_chunks = []
    all_embeddings = []
    for doc in st.session_state.documents:
        all_chunks.extend(doc["document_chunks"])
        all_embeddings.extend(doc["document_embeddings"])
    doc_embeddings = np.vstack(all_embeddings)
    similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
    top_indices = np.argsort(similarities)[-Config.TOP_N:][::-1]
    results = [(all_chunks[i], similarities[i]) for i in top_indices]
    return results

def generate_answer(user_query: str, context: str) -> str:
    prompt = (
        f"System: {Config.SYSTEM_PROMPT}\n\n"
        f"Context:\n{context}\n\n"
        f"User: {user_query}\nAssistant:"
    )
    try:
        model = palm.GenerativeModel(Config.GENERATION_MODEL)
        response = model.generate_content(prompt)
        if hasattr(response, "text"):
            return response.text
        else:
            return response
    except Exception as e:
        logger.error("Failed to generate answer: %s", e)
        st.error("Failed to generate answer. Please check your input and try again.")
        return "I'm sorry, I encountered an error generating a response."

def chat_app():
    if "conversations" not in st.session_state:
        st.session_state.conversations = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    for conv in st.session_state.conversations:
        with st.chat_message("user"):
            st.write(conv.get("user_question", ""))
        with st.chat_message("assistant"):
            st.write(conv.get("assistant_answer", ""))
            if conv.get("feedback"):
                st.markdown(f"**Feedback:** {conv['feedback']}")
    user_input = st.chat_input("Type your message here")
    if user_input:
        with st.chat_message("user"):
            st.write(user_input)
        results = search_query(user_input)
        context = "\n\n".join([chunk for chunk, score in results]) if results else ""
        answer = generate_answer(user_input, context)
        with st.chat_message("assistant"):
            st.write(answer)
        conversation_id = save_conversation_to_firestore(
            st.session_state.session_id, 
            user_question=user_input, 
            assistant_answer=answer
        )
        st.session_state.latest_conversation_id = conversation_id
        st.session_state.conversations.append({
            "user_question": user_input,
            "assistant_answer": answer,
        })
        col1, col2 ,col3,col4,col5= st.columns(5)
        col1.button("👍", key=f"feedback_like_{len(st.session_state.conversations)}", on_click=handle_feedback, args=("positive",))
        col2.button("👎", key=f"feedback_dislike_{len(st.session_state.conversations)}", on_click=handle_feedback, args=("negative",))

# Define the clear confirmation dialog using st.dialog decorator.
@st.dialog("Confirm Clear")
def clear_confirm_dialog():
    st.write("This will erase all attached documents and chat history. Do you want to proceed?")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Confirm Clear"):
            clear_documents()
            st.success("Documents and chat history have been cleared.")
            st.rerun()
    with col2:
        if st.button("Cancel"):
            st.write("Operation cancelled.")
            st.rerun()

def main():
    st.title("Chat with your files")
    st.sidebar.header("Upload Documents")
    
    # Ensure a dynamic key for the file uploader exists.
    if "uploaded_files_key" not in st.session_state:
        st.session_state["uploaded_files_key"] = str(uuid.uuid4())
    
    # File uploader using the dynamic key.
    uploaded_files = st.sidebar.file_uploader(
        "Upload (.txt, .pdf, .docx)",
        type=["txt", "pdf", "docx"],
        accept_multiple_files=True,
        key=st.session_state["uploaded_files_key"]
    )
    if uploaded_files:
        for file in uploaded_files:
            process_document(file)
    
    # Show the clear button if either documents, conversations exist or if files are uploaded.
    if (("documents" in st.session_state and st.session_state.documents) or
        ("conversations" in st.session_state and st.session_state.conversations) or
        (uploaded_files is not None and len(uploaded_files) > 0)):
        if st.sidebar.button("Clear Documents & Chat History"):
            clear_confirm_dialog()  # Call the dialog function.
    
    if st.session_state.get("doc_processed", False):
        chat_app()
    else:
        st.info("Please upload and process at least one document from the sidebar to start chatting.")
        
    st.markdown(
        """
        <div style="position: fixed; right: 10px; bottom: 10px; font-size: 12px; z-index: 9999; text-align: right;">
            Your questions, our response as well as your feedback will be saved for evaluation purposes.
        </div> 
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
