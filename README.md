# Requirements to run the project

# Download Ollama from ollama.com

# After installing Ollama, in windows CMD, type
Ollama run llama3.1
Ollama run phi3
Ollama pull nomic-embed-text 


# Python packages to install
pip install flask
pip install langchain-community
pip install langchain
pip install chromadb
pip install fastembedding
pip install PyMuPDF
pip install streamlit
pip install ollama

# microsoft Visual C++ is required for Chromadb

# after installing packages, you can manually run RAG app
# under the directory "RAG" , run the following command
python -m flask run 
# the flask server will run. You can manaull use RAG app through http requests. To get direct llm response, in browser, type
http://127.0.0.1:5000/ask?question=Your+Question+Here

# to setup UiPath, open "DocBot_UiPath" folder in UiPath.
# You will have to redirect the path to python "RAG" folder
# In main.xaml, in "Use Application: Command Prompt to run python rag server" >> replace the command to type >> with your machine directory route to "RAG" folder

# After running UiPath, the result response are stored at "AIresponses.xlsx" in the "DocBot_UiPath" folder.


# You can also run our app with GUI
# under the directory "strealitRAG", run the following command
python -m streamlit run app.py

# Have fun!