import os
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pinecone import Pinecone, ServerlessSpec

# Load environment variables
load_dotenv()
PCapi = os.getenv('ragdoctor')
gemapi = os.getenv('GemAPI')
os.environ["PINECONE_API_KEY"] = PCapi
os.environ["GOOGLE_API_KEY"] = gemapi

# Initialize HuggingFace embedding model
def Hf_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

Embedding = Hf_embedding_model()

# Pinecone setup
def checkVectorDB(name, PCapi):
    pc = Pinecone(api_key=PCapi)
    existing_indexes = pc.list_indexes()
    for idx in existing_indexes:
        if idx['name'] == name:
            return True
    return False

index_name = "doctordb"
pc = Pinecone(api_key=PCapi)

if not checkVectorDB(index_name, PCapi):
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
else:
    print("Vector database already exists.")

# Load vector store
searchdocs = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=Embedding
)

retriever = searchdocs.as_retriever(
    search_type="similarity", search_kwargs={"k": 10}
)

# Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.7,
    max_tokens=1024
)

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

# QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": custom_prompt}
)

# Query
query = "what is Atherosclerosis why it happens"
result = qa_chain.invoke(query)

# Output
print("Answer:", result['result'])
# Uncomment if you want to print source documents
# for doc in result['source_documents']:
#     print(doc.metadata.get('source', 'Unknown Source'))
