# Knowledge Base Chatbot 📚

Knowledge Base Chatbot answers all your queries.

## Description

Konwledge Base Chatbot, is a conversational document retrieval system that allows users to upload PDF documents and then interactively ask questions about the content of those documents. The system is built using Streamlit, a Python library for building web applications, and leverages several Natural Language Processing (NLP) tools and models for text processing and document retrieval.

Langchain, an open source framework for building LLM application has been used.

### Functionalities:

**1. Document Processing:** Upon uploading one or multiple PDF documents, the system extracts the text content from each document and aggregates them into a single data string.

**2. Text Chunking:** The aggregated data is then divided into smaller chunks of text using a character-based text splitter. Each chunk has a specified size with a small overlap to ensure that relevant information is retained.

**3. Text Embeddings:** The text chunks are converted into numerical embeddings using Hugging Face's Instruct Embeddings, which is a state-of-the-art NLP model capable of encoding text into dense vectors. These embeddings capture the semantic meaning of the text.

**4. Vector Store Creation:** The embeddings are used to create a vector store using the FAISS library, which efficiently indexes and organizes the embeddings for fast retrieval.

**5. Conversational Retrieval Chain:** The system uses OpenAI's GPT-3-based language model (LLM) as a conversational agent. The LLM is combined with the vector store to create a Conversational Retrieval Chain. This chain allows the system to respond contextually and relevantly to user queries by retrieving information from the document chunks based on the user's conversation history.

### Working:

**1. User Interaction:** The application presents a web interface where users can interact with the system through a chat-like interface.

**2. Document Upload:** Users can upload their PDF documents through the sidebar. Upon clicking "Process," the documents are processed, and text data is extracted.

**3. Question Asking:** Users can then ask questions about the content of the uploaded documents in natural language.

**4. Question Answering:** The system takes the user's question, and using the conversational retrieval chain, it retrieves the most relevant information from the document chunks. The response is then presented to the user in JSON format.

**5. Context Maintenance:** The system maintains conversation history to ensure that follow-up questions receive appropriate context, making the interaction more seamless and natural.

## Installation

1. Clone the repository in your local machine.
2. Activate a virtual environment in the project, so that the dependencies are installed locally and is isolated from other projets to avoid conflicts.

   **Create a virtual env**

   ```
   python3 -m venv venv
   ```

   Activating the virtual env

   **For Linux**

   ```
   source venv/bin/activate
   ```

   **For Windows**

   ```
   venv\Scripts\activate
   ```

3. Install all the requirements of the project.
   ```
   pip install -r requirements.txt
   ```
   This will automatically install all the dependencies in the virtual environment.
4. Setup an OPENAI_API_KEY in the .env file.
5. Run the application
   ```
   streamlit run app.py
   ```
## Future Developments (In progress)
1. Read data from a database, say Postgres, instead of uploading docs.
2. Feedback Mechanism, allowing user to rate the answer.
3. Image Support
4. Voice Input/Output
