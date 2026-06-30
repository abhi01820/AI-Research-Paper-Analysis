import os,json
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from flask import Flask,render_template,request,jsonify

from src.load_and_extract_text import extract_text_from_pdf,extract_pdf_sections
from src.detect_and_split_sections import refine_sections,split_sections_with_content
from src.get_summary import generate_detailed_summary
from src.create_vector_db import create_vector_db
from src.RAG_retrival_chain import get_qa_chain

from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

groq_api_key=os.getenv("GROQ_API_KEY")
embedding_model=os.getenv("EMBEDDING_MODEL")
llm_model=os.getenv("LLM_MODEL")

full_text=''
Research_paper_topics=None
vector_db=None

llm=ChatGroq(groq_api_key=groq_api_key,model_name=llm_model)
embedder=HuggingFaceEmbeddings(model_name=embedding_model)

# print(llm.invoke("who is Hardik Pandya?"))


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_pdf():
    global full_text
    global Research_paper_topics
    
    file = request.files.get('file')
    
    if not file:
        return jsonify({"error": "No file uploaded"}), 400
    
    filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filename)
    
    extracted_text = extract_text_from_pdf(filename)
    full_text = extracted_text
    extracted_sections = extract_pdf_sections(full_text=extracted_text)

    refined_sections = refine_sections(extracted_sections, llm)

    section_with_content = split_sections_with_content(extracted_text, refined_sections)
    
    Research_paper_topics = section_with_content
    
    return jsonify({"topics": list(Research_paper_topics.keys())})


@app.route('/summary', methods=['POST'])
def get_summary():
    global Research_paper_topics
    
    topic = request.json.get('topic')
    
    topic_content = Research_paper_topics.get(topic, "No summary available.")
    
    summary = generate_detailed_summary(topic_content, llm)
    
    return jsonify({"summary": summary})
    

@app.route('/chat', methods=['POST'])
def chat():
    global full_text
    global vector_db
    
    user_message = request.json.get('message')
    print(user_message)
    
    if not vector_db:
        vectordb = create_vector_db(text=full_text, embedder=embedder)
        vector_db = vectordb
        
    chain = get_qa_chain(vectordb=vector_db, llm=llm)
    
    ai_response = chain.invoke(user_message)['result']
    print(ai_response)
    
    return jsonify({"response": ai_response})
        
    
    
    



if __name__=="__main__":
    app.run(debug=True)