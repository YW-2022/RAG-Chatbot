import streamlit as st
from openai import AzureOpenAI
from os import environ

# Set up the title of the Streamlit app
st.title("📝 Chat with txt files")

# Create a file uploader widget that accepts .txt files
uploaded_file = st.file_uploader("Upload a txt file", type=("txt"))

# Create an input field for user questions, disabled until a file is uploaded
question = st.chat_input(
    "Ask something about the txt file",
    disabled=not uploaded_file,
)

# Initialize the 'messages' state to store chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Ask something about the txt file"}
    ]

# Display chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Proceed if the user inputs a question and a file is uploaded
if question and uploaded_file:
    # Read the content of the uploaded .txt file and decode it as a string
    file_content = uploaded_file.read().decode("utf-8")
    print(file_content)  # Print the content for debugging (optional)

    # Set up the Azure OpenAI client using environment variables for credentials
    client = AzureOpenAI(
        api_key=environ['AZURE_OPENAI_API_KEY'],
        api_version="2023-03-15-preview",
        azure_endpoint=environ['AZURE_OPENAI_ENDPOINT'],
        azure_deployment=environ['AZURE_OPENAI_MODEL_DEPLOYMENT'],
    )

    # Add the user's question to the chat history in session state
    st.session_state.messages.append({"role": "user", "content": question})
    st.chat_message("user").write(question)  # Display the user's question

    # Generate a response from the assistant using the Azure OpenAI client
    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model="gpt-4o",  # Change this to a valid model name
            messages=[
                {"role": "system", "content": f"Here's the content of the file:\n\n{file_content}"},
                *st.session_state.messages
            ],
            stream=True  # Enable streaming of the response
        )
        response = st.write_stream(stream)  # Write the streamed response

    # Add the assistant's response to the chat history in session state
    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )
