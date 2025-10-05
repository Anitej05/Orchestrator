# agents/document_analysis_agent.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_cerebras import ChatCerebras
import os
from dotenv import load_dotenv
import logging

# Load environment variables from a .env file at the project root
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration & API Key Check ---
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")
if not CEREBRAS_API_KEY:
    raise RuntimeError("CEREBRAS_API_KEY is not set in the environment. The agent cannot start.")

# --- Model and Embeddings Loading ---
# Initialize the embedding model. This runs locally to convert text chunks into vectors.
# 'all-mpnet-base-v2' is a high-performance model, great for semantic search.
hf_embeddings = HuggingFaceEmbeddings(model_name='all-mpnet-base-v2')

# Initialize the Groq Chat client for the generation step of the RAG pipeline.
llm = ChatCerebras(model="gpt-oss-120b")

# --- FastAPI Application ---
app = FastAPI(
    title="Document Analysis Agent",
    description="A RAG-based agent that answers questions about documents.",
    version="1.0.0"
)

# --- Pydantic Models for API Data Validation ---
class AnalyzeDocumentRequest(BaseModel):
    """Defines the expected input for the /analyze endpoint."""
    vector_store_path: str = Field(..., description="The file path to the FAISS vector store index for the document.")
    query: str = Field(..., description="The user's question about the document.")

class AnalyzeDocumentResponse(BaseModel):
    """Defines the output format for the /analyze endpoint."""
    answer: str

# --- API Endpoints ---
@app.get("/", summary="Health Check")
def read_root():
    """Provides a simple health check endpoint."""
    return {"message": "Document Analysis Agent is running and ready to analyze documents."}

@app.post("/analyze",
          response_model=AnalyzeDocumentResponse,
          summary="Analyze a Document via RAG")
def analyze_document(request: AnalyzeDocumentRequest):
    """
    Analyzes a document using a RAG pipeline. It loads a pre-computed FAISS vector store,
    retrieves relevant text chunks based on the user's query, and then uses a Groq LLM
    to generate a final answer from that context.
    """
    # 1. Validate that the vector store path exists.
    if not os.path.exists(request.vector_store_path):
        logger.error(f"Document vector store not found at path: {request.vector_store_path}")
        raise HTTPException(
            status_code=404,
            detail=f"Document vector store not found at path: {request.vector_store_path}"
        )

    try:
        # 2. Load the vector store from the path provided by the orchestrator.
        # This is the core of the RAG pipeline's retrieval step.
        # The `allow_dangerous_deserialization` flag is required for FAISS with pickle.
        logger.info(f"Loading vector store from: {request.vector_store_path}")
        vector_store = FAISS.load_local(
            request.vector_store_path,
            hf_embeddings,
            allow_dangerous_deserialization=True
        )

        # 3. Create a RetrievalQA chain, a standard LangChain component for RAG.
        # It automates retrieving relevant documents and passing them to the LLM.
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",  # "stuff" chain type includes all retrieved text in the prompt.
            retriever=vector_store.as_retriever(search_kwargs={"k": 5}) # Retrieve top 5 chunks
        )

        # 4. Invoke the chain to get the answer. This performs the retrieval and generation steps.
        logger.info(f"Processing query: {request.query}")
        result = qa_chain.invoke({"query": request.query})

        # 5. Return the generated answer in the specified response format.
        logger.info("Document analysis completed successfully")
        return AnalyzeDocumentResponse(answer=result['result'])

    except Exception as e:
        # Catch-all for any other errors during the process
        logger.error(f"Error occurred while analyzing the document: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred while analyzing the document: {str(e)}"
        )

if __name__ == "__main__":
    # This block allows you to run the server directly for testing
    # Use: python backend/agents/document_analysis_agent.py
    port = int(os.getenv("DOCUMENT_AGENT_PORT", 8070))
    logger.info(f"Starting Document Analysis Agent on port {port}")
    uvicorn.run("document_analysis_agent:app", host="127.0.0.1", port=port, reload=True)
