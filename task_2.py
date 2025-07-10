from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

# Load and split PDF
loader = PyMuPDFLoader("/home/ahmad/Python/Practice_set/Application.pdf")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)

# Create embedding + FAISS vector store
embedding = OllamaEmbeddings(model="llama3")
vectorstore = FAISS.from_documents(chunks, embedding)

# Build retriever and RAG chain
retriever = vectorstore.as_retriever()
llm = Ollama(model="llama3")

rag = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Query example
query = "What is this PDF about?"
result = rag.run(query)
print("Answer:", result)

# import sys # Import sys for flushing output

# print("Script started.", file=sys.stderr, flush=True) # Print to stderr, flush immediately

# try:
#     from langchain_ollama import OllamaEmbeddings
#     print("OllamaEmbeddings imported.", file=sys.stderr, flush=True)
# except Exception as e:
#     print(f"Error importing OllamaEmbeddings: {e}", file=sys.stderr, flush=True)
#     sys.exit(1) # Exit if import fails

# try:
#     from langchain_community.llms import Ollama
#     print("Ollama LLM imported.", file=sys.stderr, flush=True)
# except Exception as e:
#     print(f"Error importing Ollama LLM: {e}", file=sys.stderr, flush=True)
#     sys.exit(1)

# # Ensure all other necessary imports are here and correct
# try:
#     from langchain_community.document_loaders import PyMuPDFLoader
#     from langchain.text_splitter import RecursiveCharacterTextSplitter
#     from langchain_community.vectorstores import FAISS
#     from langchain.chains import RetrievalQA
#     print("All other LangChain modules imported.", file=sys.stderr, flush=True)
# except Exception as e:
#     print(f"Error importing other LangChain modules: {e}", file=sys.stderr, flush=True)
#     sys.exit(1)


# # Load and split PDF
# try:
#     # Use full path as you have it, ensure the file exists at this exact path.
#     pdf_path = "/home/ahmad/Python/Practice_set/AhmadRaza.pdf"
#     print(f"Attempting to load PDF from: {pdf_path}", file=sys.stderr, flush=True)
#     loader = PyMuPDFLoader(pdf_path)
#     docs = loader.load()
#     print(f"PDF loaded. Number of pages/docs: {len(docs)}", file=sys.stderr, flush=True)
# except Exception as e:
#     print(f"Error loading PDF: {e}", file=sys.stderr, flush=True)
#     sys.exit(1)


# # Split documents
# try:
#     splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#     chunks = splitter.split_documents(docs)
#     print(f"Documents split into {len(chunks)} chunks.", file=sys.stderr, flush=True)
# except Exception as e:
#     print(f"Error splitting documents: {e}", file=sys.stderr, flush=True)
#     sys.exit(1)


# # Create embedding + FAISS vector store
# try:
#     print("Initializing OllamaEmbeddings...", file=sys.stderr, flush=True)
#     embedding = OllamaEmbeddings(model="llama3")
#     print("Creating FAISS vector store...", file=sys.stderr, flush=True)
#     vectorstore = FAISS.from_documents(chunks, embedding)
#     print("Vector store created.", file=sys.stderr, flush=True)
# except Exception as e:
#     print(f"Error creating embeddings or vector store: {e}", file=sys.stderr, flush=True)
#     sys.exit(1)


# # Build retriever and RAG chain
# try:
#     print("Building RAG chain...", file=sys.stderr, flush=True)
#     retriever = vectorstore.as_retriever()
#     llm = Ollama(model="llama3")
#     rag = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
#     print("RAG chain built.", file=sys.stderr, flush=True)
# except Exception as e:
#     print(f"Error building RAG chain: {e}", file=sys.stderr, flush=True)
#     sys.exit(1)


# # Query example
# try:
#     query = "What is this PDF about?"
#     print(f"Querying LLM: '{query}'", file=sys.stderr, flush=True)
#     result = rag.invoke({"query": query})
#     print("Query completed.", file=sys.stderr, flush=True)
#     print("Answer:", result['result'], flush=True) # This prints to stdout
# except Exception as e:
#     print(f"Error during query: {e}", file=sys.stderr, flush=True)
#     sys.exit(1)

# print("Script finished.", file=sys.stderr, flush=True)