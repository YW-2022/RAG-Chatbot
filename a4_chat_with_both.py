from langchain_text_splitters import RecursiveCharacterTextSplitter
import streamlit as st
from openai import AzureOpenAI
import os
from langchain_community.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain_openai import AzureOpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import AzureChatOpenAI
import chromadb.api

# Set up the title for the Streamlit app
st.title("📝 RAG Chatbot")

# Create a file uploader for .txt and .pdf files
uploaded_file = st.file_uploader("Upload a txt file", type=("txt", "pdf"))

# Initialize the chat history in the session state
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Hello! How can I help you today?"}]

# Display previous chat messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Create an input field for user questions
prompt1 = st.chat_input()

# Check if a file has been uploaded
if uploaded_file:
    st.write("File has been loaded successfully!")

    # Determine the file extension to handle .txt and .pdf files appropriately
    file_extension = os.path.splitext(uploaded_file.name)[1].lower()

    # Process .txt files
    if file_extension == ".txt":
        with open("uploaded_temp.txt", "wb") as temp_file:
            temp_file.write(uploaded_file.getbuffer())
        loader = TextLoader("uploaded_temp.txt")

    # Process .pdf files
    elif file_extension == ".pdf":
        temp_file = "./temp.pdf"
        with open(temp_file, "wb") as file:
            file.write(uploaded_file.getvalue())
            file_name = uploaded_file.name
        loader = PyPDFLoader(temp_file)

    # Load the document content using the appropriate loader
    documents = loader.load()

    # Split the loaded document into smaller chunks for processing
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,  # Size of each chunk
        chunk_overlap=10,  # Overlap between chunks to ensure context continuity
        separators=["\n\n", "\n", " ", ""]  # Separators for splitting the text
    )
    chunks = text_splitter.split_documents(documents)

    # Create a vector store from the document chunks using embeddings
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=AzureOpenAIEmbeddings(model="text-embedding-3-large")
    )

    # Clear cache to ensure updated data handling in ChromaDB
    chromadb.api.client.SharedSystemClient.clear_system_cache()

    # Define a template for generating responses with context
    template = """
       You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.
       
       Question: {question} 
       
       Context: {context} 
       
       Answer:
    """
    prompt = PromptTemplate.from_template(template)

    # Set up a retriever to fetch relevant context for the user's question
    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 1}  # Retrieve the most similar chunk
    )

    # Format documents for better readability in responses
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    # Initialize the Azure Chat OpenAI model for generating responses
    llm = AzureChatOpenAI(
        azure_deployment="gpt-4o",
        temperature=0.2,  # Lower temperature for more focused responses
        api_version="2023-06-01-preview",
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )

    # Create a RAG (Retrieval Augmented Generation) chain that combines context retrieval and response generation
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

# Handle user input when a prompt is provided
if prompt1:
    # Initialize the Azure OpenAI client for generating responses
    client = AzureOpenAI(
        azure_deployment="gpt-4o",
        api_version="2023-06-01-preview"
    )

    # Add the user's prompt to the session state chat history
    st.session_state.messages.append({"role": "user", "content": prompt1})
    st.chat_message("user").write(prompt1)

    # Generate a response using the RAG chain if a file is uploaded
    if uploaded_file:
        response = rag_chain.invoke(prompt1)
        st.chat_message("assistant").write(response)
    else:
        # Generate a response directly from the Azure OpenAI model if no file is uploaded
        with st.chat_message("assistant"):
            stream = client.chat.completions.create(
                model="gpt-4o", 
                messages=st.session_state.messages,
                stream=True  # Enable streaming for faster response
            )
            response = st.write_stream(stream)

    # Add the assistant's response to the chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
