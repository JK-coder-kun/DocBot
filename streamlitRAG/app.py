import streamlit as st
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import PromptTemplate
import pymupdf

vector_store = None

# LLM Model Selection
st.title("Document Search and AI Assistant")

st.sidebar.title("LLM Model Selection")
selected_model = st.sidebar.selectbox(
    "Choose an LLM model", 
    ["llama3.1", "phi3"]
)

# Initialize LLM based on selection
if selected_model == "llama3.1":
    cached_llm = Ollama(model="llama3.1")
elif selected_model == "phi3":
    cached_llm = Ollama(model="phi3")

# Initialize Embedding model
embedding = FastEmbedEmbeddings()

# Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=128)

# Default Prompt Template
default_prompt = "Answer the question straight forward and shortly."
default_prompt_template = f""" 
<s>[INST]{default_prompt}  [/INST] </s>
[INST] {{input}}
       Context: {{context}}
       Answer:
[/INST]
"""


# UI for editing the PromptTemplate
st.sidebar.title("Prompt Template Editor")
edited_prompt = st.sidebar.text_area(
    "Edit the Prompt Template", 
    default_prompt, 
    height=300
)

edited_prompt_template = f""" 
<s>[INST]{edited_prompt}  [/INST] </s>
[INST] {{input}}
       Context: {{context}}
       Answer:
[/INST]
"""

# Apply the user's edited template
try:
    raw_prompt = PromptTemplate.from_template(edited_prompt_template)
except Exception as e:
    st.sidebar.error(f"Error in template: {e}")
    raw_prompt = PromptTemplate.from_template(default_prompt_template)

# Streamlit UI for asking questions
st.header("Ask the AI Assistant")

question = st.text_input("Enter your question:")
if st.button("Ask AI"):
    if question:
        st.write(f"**Question**: {question}")
        # Call the LLM to generate a response
        response = cached_llm.invoke(question)
        st.write(f"**Response**: {response}")

# Upload PDF and Process Section
st.header("Upload a PDF to Create a Searchable Document")

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    save_file = f"pdf/{uploaded_file.name}"
    
    # Save the uploaded file
    with open(save_file, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"Uploaded {uploaded_file.name}")

    # Load and process the PDF
    docs = pymupdf.open(save_file)
    st.write(f"Number of pages loaded: {len(docs)}")
    text_document = "\n\n".join([page.get_text() for page in docs])

    # Split documents into chunks
    chunks = text_splitter.split_text(text_document)
    st.write(f"Number of chunks created: {len(chunks)}")

    # Store document chunks in the vector store
    vector_store = Chroma.from_texts(texts=chunks, embedding=embedding)
    st.success("Document has been processed and saved!")

    # Load vector store
    retriever = vector_store.as_retriever()

    # Create chain for retrieval and document processing
    document_chain = create_stuff_documents_chain(cached_llm, raw_prompt)
    chain = create_retrieval_chain(retriever, document_chain)

# Query the PDF content
st.header("Ask Questions from the Uploaded PDF")

query = st.text_input("Enter your query to search the document:")
if st.button("Ask PDF"):
    if query and uploaded_file:
        # Retrieve and generate answer
        result = chain.invoke({"input": query})
        
        # Display result
        st.write(f"**Answer**: {result['answer']}")
    else:
        st.write("Reupload Document and try again!")
