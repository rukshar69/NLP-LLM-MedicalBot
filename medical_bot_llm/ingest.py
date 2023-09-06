# Import necessary modules from the langchain library
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter 

# Define the path to the data directory and the path to save the vector database
DATA_PATH = 'data/'
DB_FAISS_PATH = 'vectorstore/db_faiss'

# Define a function to create a vector database
"""
This is a Python script that creates a vector database using the langchain library. The script imports several modules from the langchain library, including HuggingFaceEmbeddings, FAISS, PyPDFLoader, DirectoryLoader, and RecursiveCharacterTextSplitter.

The script defines a function called create_vector_db() that creates a vector database. The function does the following:

It creates an instance of the DirectoryLoader class, which loads all PDF files from the specified data path using the PyPDFLoader class.
It loads the documents and splits them into smaller chunks using an instance of the RecursiveCharacterTextSplitter class.
It creates an instance of the HuggingFaceEmbeddings class, which is used to generate embeddings for the text chunks.
It creates an instance of the FAISS class, which is used to create a vector database from the text chunks and their embeddings.
It saves the vector database to a local file specified by the DB_FAISS_PATH variable.
"""
def create_vector_db():
    # Create an instance of the DirectoryLoader class to load PDF files from the data directory
    loader = DirectoryLoader(DATA_PATH,
                             glob='*.pdf',
                             loader_cls=PyPDFLoader)

    # Load the documents and split them into smaller chunks using an instance of the RecursiveCharacterTextSplitter class
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                                   chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    # Create an instance of the HuggingFaceEmbeddings class to generate embeddings for the text chunks
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cpu'})

    # Create an instance of the FAISS class to create a vector database from the text chunks and their embeddings
    db = FAISS.from_documents(texts, embeddings)
    
    # Save the vector database to a local file
    db.save_local(DB_FAISS_PATH)

# Check if the script is being run as the main program and call the create_vector_db() function if it is
if __name__ == "__main__":
    create_vector_db()

