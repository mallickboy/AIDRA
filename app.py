from flask import Flask, render_template, request
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
from pinecone import Pinecone

load_dotenv()

app = Flask(__name__)


embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

PCapi = os.getenv('ragdoctor')
os.environ["PINECONE_API_KEY"] = PCapi
pc = Pinecone(api_key=PCapi)
index_name = "doctordb"

# Loading PC
vector_store = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embedding_model)
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 10})

# Gemini init
gemapi = os.getenv('GemAPI')
os.environ["GOOGLE_API_KEY"] = gemapi
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7, max_tokens=1024)

# Custom prompt
custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are an expert in holistic and alternative medicine, with deep knowledge sourced from trusted resources like *The Gale Encyclopedia of Alternative Medicine*.

Using the following context from the encyclopedia, answer the question thoughtfully and factually. Focus on natural remedies, therapies, and traditional practices when relevant.

If the answer cannot be found in the context, respond with: "The provided encyclopedia content does not contain a definitive answer."
Use a calm, educational tone.

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
    answer = None
    question = ""
    if request.method == 'POST':
        question = request.form['question']
        result = qa_chain.invoke(question)
        answer = result['result']
    return render_template('index.html', answer=answer, question=question)

if __name__ == '__main__':
    app.run(debug=True)
