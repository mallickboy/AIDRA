# qa_pipeline.py

import os
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pinecone import Pinecone

load_dotenv()

def initialize_qa_chain():
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    PCapi = os.getenv('ragdoctor')
    index_name = "doctordb"
    pc = Pinecone(api_key=PCapi)

    def checkVectorDB(name):
        existing_indexes = pc.list_indexes()
        return any(i['name'] == name for i in existing_indexes)

    if not checkVectorDB(index_name):
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec={"cloud": "aws", "region": "us-east-1"}
        )

    os.environ["PINECONE_API_KEY"] = PCapi
    vectorstore = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embedding)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 30})

    custom_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are an expert in holistic and alternative medicine...

Context:
{context}

Question:
{question}

Answer:
"""
    )

    gemapi = os.getenv("GemAPI")
    os.environ["GOOGLE_API_KEY"] = gemapi
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7, max_tokens=1024)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": custom_prompt}
    )

    return qa_chain

qa_chain = initialize_qa_chain()

def get_answer(query):
    result = qa_chain.invoke(query)
    return result["result"]
