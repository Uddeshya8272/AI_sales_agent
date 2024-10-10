import os
import warnings
from langchain.llms import BaseLLM 
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

warnings.filterwarnings("ignore")

# Add this class in your rag_with_gemini.py
from langchain.llms import BaseLLM  

class RAGPipelineWrapper(BaseLLM):
    def __init__(self, rag_pipeline):
        self.rag_pipeline = rag_pipeline

    def _call(self, prompt: str) -> str:
        return self.rag_pipeline.run(prompt)


class RAGPipeline:
    def __init__(self, file_path, google_api_key):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file at {file_path} does not exist.")
        self.google_api_key = google_api_key
        self.file_path = file_path

        # Initialize LLM (Gemini 1.5 Pro)
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=google_api_key,
            temperature=0.4,
            max_tokens=None,
        )

        # Load CSV data with error handling
        try:
            self.csv_loader = CSVLoader(file_path=self.file_path)
            pages = self.csv_loader.load()
        except Exception as e:
            raise RuntimeError(f"Failed to load the CSV file: {str(e)}")

        # Text splitting with reduced chunk size and overlap
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        context = "\n\n".join(str(p.page_content) for p in pages)
        texts = self.text_splitter.split_text(context)

        # Embedding creation and vector storage
        try:
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=self.google_api_key
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize embeddings: {str(e)}")

        self.vector_index = Chroma.from_texts(texts, self.embeddings).as_retriever(search_kwargs={"k": 1})

        # Set up the Retrieval-based QA Chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.vector_index,
            return_source_documents=True
        )

    def ask_question(self, question):
        try:
            result = self.qa_chain({"query": question})
            return result.get("result", "No result found")
        except Exception as e:
            return f"Error in retrieving answer: {str(e)}"

    def query_product_info(self, product_name):
        question = f"Tell me more about the product: {product_name}"
        return self.ask_question(question)
