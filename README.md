# Chat with Your Files

A Streamlit-based web application that allows users to upload and chat with text-based documents (.txt, .pdf, .docx). The app extracts, processes, and chunks document text for retrieval-augmented question answering.

## Features

- Upload and process `.txt`, `.pdf`, and `.docx` documents.
- Store and retrieve messages using Firebase Firestore.
- Generate embeddings using Google's Gemini model.
- Perform document similarity searches with cosine similarity.
- Chat interface powered by Streamlit.
- Store user feedback for system evaluation.

## Requirements

Ensure you have the following dependencies installed:

```
pip install streamlit pdfplumber python-docx firebase-admin scikit-learn numpy google-generativeai
```

## Environment Variables

The application requires environment variables for Firebase credentials and Google API key:

```
FIREBASE_CRED=<Your Firebase JSON credentials>
GOOGLE_API_KEY=<Your Google API Key>
```

## How It Works

1. **Upload Documents**: Users upload `.txt`, `.pdf`, or `.docx` files.
2. **Text Extraction**: The app extracts and processes text using `pdfplumber` and `python-docx`.
3. **Chunking & Embeddings**: Text is divided into smaller chunks, and embeddings are generated using Google's Gemini model.
4. **Chat with Documents**: Users can ask questions, and the system retrieves relevant document chunks using cosine similarity and generates responses.
5. **Store Conversations**: The chat history and feedback are stored in Firebase Firestore.

## Installation & Usage

### Clone the Repository
```
git clone <repository-url>
cd <repository-folder>
```

### Run the Application
```
streamlit run app.py
```

## Firebase Setup
Ensure that you have Firebase Firestore enabled. Store your Firebase credentials in an environment variable named `FIREBASE_CRED`.

## Key Functionalities

### Extracting Text from Files
- Extract text from PDF, DOCX, and TXT files.
- Process extracted text into manageable chunks.

### Embedding Generation
- Uses Google Gemini model to generate embeddings.
- Implements caching for efficiency.

### Document Retrieval
- Uses cosine similarity to find the most relevant document sections.
- Returns top `N` relevant results.

### Chatbot Interaction
- Provides a chat interface for user interaction.
- Stores conversation history in Firebase Firestore.

### Feedback Mechanism
- Users can provide feedback on generated responses.
- Feedback is stored for analysis and model improvement.

## Clearing Data
- Users can clear uploaded documents and chat history via a confirmation dialog.

## Technologies Used

- **Python**
- **Streamlit**
- **Firebase Firestore**
- **Google Generative AI (Gemini)**
- **Scikit-Learn (cosine similarity)**
- **PDFPlumber & Python-docx**

## Contributing

1. Fork the repository.
2. Create a new branch: `git checkout -b feature-branch`
3. Commit your changes: `git commit -m 'Add new feature'`
4. Push the branch: `git push origin feature-branch`
5. Open a pull request.

## License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

- Inspired by AI-powered document analysis.
- Uses Google's Gemini API for embedding and content generation.
- Powered by Firebase Firestore for data storage.

---
**Note:** User questions, responses, and feedback are stored for evaluation purposes.

