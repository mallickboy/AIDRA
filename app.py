from flask import Flask, render_template, jsonify, request
# from src.helper import download_hugging_face_embeddings
# from src.healper import 
# from langchain.chains import create_retrival_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate

from pinecone import Pinecone, ServerlessSpec
import os
from dotenv import load_dotenv

from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from src.custom_prompt import *
from utility import *
import os



app = Flask(__name__)

load_dotenv()
PCapi = os.getenv('ragdoctor')
gemapi= os.getenv('GemAPI')

os.environ["PINECONE_API_KEY"] = PCapi
os.environ["GOOGLE_API_KEY"] = gemapi

Embedding = Hf_embedding_model()

index_name = 'doctordb'

pc = Pinecone(api_key=PCapi)
index_name = "doctordb"
if not checkVectorDB(index_name,PCapi):
    pc.create_index(
        name=index_name,
        dimension=384, #  model dimensions
        metric="cosine", #model metric
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
else:
    print("database created Previously!")

searchdocs = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=Embedding
)

retriver = searchdocs.as_retriever(search_type="similarity", search_kwargs={"k":10})

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",  # or "gemini-1.5-flash"
    temperature=0.7,
    max_tokens=1024
)

custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=custom_prompt_template
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriver,
    return_source_documents=True,
    chain_type_kwargs={"prompt": custom_prompt}
)

question_answer_chain= create_stuff_documents_chain(llm, custom_prompt)
# rag_chain = create_retrival_chain(retriver, question_answer_chain)

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    print("User input:", msg)
    response = qa_chain.invoke({"question": msg})
    print("Response:", response.get("answer", "No answer found."))
    return str(response.get("answer", "No answer found."))

if __name__ == '__main__':
    app.run(host="0.0.0.0", port = 8080, debug = True)