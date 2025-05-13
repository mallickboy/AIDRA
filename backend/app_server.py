from flask import Flask, render_template, request , session, jsonify
# from qa_pipeline import get_answer
from flask_session import Session
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
from pinecone import Pinecone
import markdown
from flask_cors import CORS
import traceback


index_name = "maindb"

load_dotenv()

app = Flask(__name__)
CORS(app)
app.secret_key = os.getenv('FLASK_SECRET_KEY')
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

PCapi = os.getenv('ragdoctor')
os.environ["PINECONE_API_KEY"] = PCapi
pc = Pinecone(api_key=PCapi)

# Loading PC
vector_store = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embedding_model)
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 50})

# Gemini init
gemapi = os.getenv('GemAPI')
os.environ["GOOGLE_API_KEY"] = gemapi
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7, max_tokens=1024)

# Custom prompt
custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a highly knowledgeable and trustworthy MBBS Doctor, Your name is MedGuide AI, specializing in  modern medicine. You are provided context from some of the world’s most authoritative clinical resources, including the Oxford Handbook of Clinical Medicine, Harrison’s Principles of Internal Medicine, The Merck Manual, and other standard medical encyclopedias .

When Users ask/ describe their symptoms, ask questions about diseases like "eg: tell me if you have these symptomps" , Tell patient to do related (desease matches to the symptom) tests if necessary "eg: You need to provide CT scan report/ blood test report.. do these tests" . Once you are sure about the desease You give medication (medicine name )

So Follow like this:
- Initially give a brief description about that desease or health condition
- Make sure patient have this particular desease (think these might be the possible deseases) if not sure tell them to provide more symptoms related to the desease , tell to do medical tests if Needed 
- Once sure about the desease, tell that he got this disease (mention name), If not sure tell them to Do x, y, z tests and give the test result,
- give  Prescription like profesional doctor when confirmed about the desease, (provide from the  sources You are already given, If you dont get source text tell "sorry No Context Available")
- Some general recommendations (eg: diet , lifestyle etc ) related to that desease 
DO it Like a Doctor 
Always act like a professional and empathetic doctor. Never invent or hallucinate treatments.If ask question outside the field give rude reply!! Only make suggestions based on the context given to you from trusted medical literature.

⚠️ Be creative in your response, but strictly grounded in the provided medical context.
Context:
{context}

Question:
{question}

Answer:
"""
)

# QA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": custom_prompt}
)

@app.route('/', methods=['GET', 'POST']) 
def index():
    try:

        if "history" not in session:
            session["history"] = []

        if request.method == 'POST':
            question = request.form.get('question', '').strip()
            if not question:
                return jsonify({"error": "No question provided"}), 400

            temp = qa_chain.invoke(question)  

            answer = temp['result']
            answer = markdown.markdown(str(answer)) if answer is not None else answer

            clean_context = [
            doc.page_content[:600]  # Truncate each document to 800 characters
            for doc in temp['source_documents'][-2:]  # Limit to the last 3 documents
        ]

            # answer = "Hi this is answer"
            # clean_context = "This is context"

            session["history"].append({
                "question": question,
                "answer": answer,
                "context": clean_context
            })
            session["history"] = session["history"][-2:] # overflow gives error
            session.modified = True  
            res= jsonify({
            'html': render_template('entry.html', question=question, answer=answer)
        }), 200
            # print(f"\n\nLOG: response send :{res}")
            return res

        return render_template('index.html'), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500




@app.route('/clear')
def clear():
    session.pop("history", None)
    return render_template("index.html", history=[], question="")

if __name__ == '__main__':
    # app.run(debug=True)
    app.run(host="0.0.0.0", port=9000)
