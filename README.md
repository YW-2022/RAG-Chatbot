# RAG-Chatbot

**Download this repository:**
```
git clone https://github.com/YW-2022/RAG-Chatbot.git
```

**Open this project in Docker container**
```
docker start info-5940-devcontainer
```
**Verify that the container is running by using:**
```
docker ps
```

**Set up environment:**

In docker-compose.yml file, change the AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT to your own API Key.
```
AZURE_OPENAI_API_KEY: xxx
AZURE_OPENAI_ENDPOINT: xxx
```

There are four main py files here:

1. a2_chat_with_txt.py : Implement functionality that allows users to upload text files with a .txt extension.

2. a3_rag_chain.py : Create a chat interface where users can ask questions about the uploaded document(s) and receive relevant answers.

3. a4_chat_with_both.py : Extend the file upload functionality to accept both .txt and .pdf files.

4. a5_chat_multiple.py : Allow users to upload multiple documents and interact with all of them within the chat interface.


To run these files:
```
streamlit run xxx.py
```

