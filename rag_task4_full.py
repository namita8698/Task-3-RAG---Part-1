# Imports
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
import os

# --- Load PDFs from folder ---
docs_folder = r"C:\Users\BUBBU\OneDrive\Desktop\Task-03-Rag-Basic\documents"
docs = []

for file_name in os.listdir(docs_folder):
    if file_name.lower().endswith(".pdf"):
        file_path = os.path.join(docs_folder, file_name)
        loader = PyPDFLoader(file_path)
        docs.extend(loader.load())

print(f"Loaded {len(docs)} document pages from PDFs.")

# --- Create embeddings ---
embeddings = OpenAIEmbeddings()  # Uses OPENAI_API_KEY

# --- Create FAISS vectorstore ---
vectorstore = FAISS.from_documents(docs, embeddings)

print("Vectorstore created successfully.")

# --- Optional: query example ---
chat_model = ChatOpenAI(model_name="gpt-3.5-turbo")
query = "Summarize the campaign guidelines."
docs_with_scores = vectorstore.similarity_search(query, k=3)

for i, doc in enumerate(docs_with_scores, start=1):
    print(f"\nResult {i}:\n{doc.page_content}")


from langchain.chains import RetrievalQA

# Create a retriever from your FAISS vectorstore
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})  # k = top 2 relevant docs

# Create a QA chain using ChatOpenAI
qa_chain = RetrievalQA.from_chain_type(
    llm=chat_model,          # your ChatOpenAI instance
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# Example query
query = "Summarize the campaign guidelines."
result = qa_chain.run(query)

print("Answer:", result)
