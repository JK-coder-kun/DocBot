from flask import Flask, request, jsonify
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
import pymupdf

# Initialize Flask app
app = Flask(__name__)

# Set up the LLM and embedding model
llm = Ollama(model="llama3.1", base_url="http://127.0.0.1:11434")
embed_model = OllamaEmbeddings(model="nomic-embed-text", base_url="http://127.0.0.1:11434")

retrieval_chain = None

def embed_document(filepath):
    # Load document and split into chunks
    document = pymupdf.open('./pdf/guide_to_online_reviews.pdf')
    document="\n\n".join([page.get_text() for page in document])
    # print("This is document------------>",document)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=128)
    chunks = text_splitter.split_text(document)
    # print("This is chunks-------------->",chunks)

    # Create a vector store and retriever
    vector_store = Chroma.from_texts(chunks, embed_model)
    return vector_store

# Flask route to handle incoming requests
@app.route('/add', methods=['GET'])
def create_chain():
    global retrieval_chain
    filepath = request.args.get("filepath", "")
    vector_store = embed_document(filepath)
    retriever = vector_store.as_retriever()
    # Formatting responses
    template = PromptTemplate(
        input_variables=["context", "input"],
        template="""
        You work at 'CellFee' online shop. Reply customers' reviews by answering not longer than 2 sentences. CellFee's email: cellfee@test.com.phone number: 08-345-221

        Context: {context}

        Question: {input}

        Answer:"""
    )

    # Create the retrieval chain
    combine_docs_chain = create_stuff_documents_chain(
        llm, template
    )
    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
    return "success"
    

@app.route('/askdoc', methods=['GET'])
def ask_doc():
    question = request.args.get("question", "")
    if not question:
        return jsonify({"error": "Please provide a question."}), 400

    try:
        # Use the retrieval chain to process the question
        response = retrieval_chain.invoke({"input": question})
        answer = response['answer']
        return answer
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/ask', methods=['GET'])
def ask():
    question = request.args.get("question", "")
    if not question:
        return jsonify({"error": "Please provide a question."}), 400

    try:
        # Use the retrieval chain to process the question
        answer = llm.invoke(question)
        return answer
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

#Sample query
# curl "http://127.0.0.1:5000/ask?question=What+is+the+story+about%3F"

