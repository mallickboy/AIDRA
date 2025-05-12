from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec

#extract data from pdf
def load_pdf_data(path):
  loader = DirectoryLoader(path, glob="*.pdf", loader_cls=PyPDFLoader)
  documents = loader.load()
  return documents

#split text into chunks
def text_splitter(extracted_data):
  text_splitter = RecursiveCharacterTextSplitter(
      chunk_size = 500,
      chunk_overlap  = 50,
      length_function = len,
  )
  text_chunk= text_splitter.split_documents(extracted_data)
  return text_chunk

# checking if the name is present or not
def checkVectorDB(name,PCapi):
    pc = Pinecone(api_key=PCapi)
    existing_indexes = pc.list_indexes()
    for i in range(len(existing_indexes)-1):
        if existing_indexes[0]['name'] == name:
            return True
    else:
        return False

# download Embedding from hugging face
def Hf_embedding_model():
  embeddings= HuggingFaceEmbeddings(model_name= "sentence-transformers/all-MiniLM-L6-v2")
  return embeddings
