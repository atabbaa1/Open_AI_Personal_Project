"""
The steps for using Langchain on private, external information are as follows:
1) Load the data using DataLoaders. This will generate Documents
2) Split the large Documents into smaller chunks using Text Splitters
3) Store the splitted data using Embeddings and Vector Stores
4) Given a user input, retrieve the relevant splits from storage using a Splitter
5) A ChatModel/ LLM generates a response using prompts
"""

# Imports
from dotenv import load_dotenv, find_dotenv
from langchain.llms import OpenAI
from langchain.schema import (AIMessage, HumanMessage, SystemMessage)
from langchain.indexes import VectorstoreIndexCreator
from langchain.prompts import PromptTemplate

load_dotenv(find_dotenv())

OPENAI_API_KEY = "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"
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
The below lines are for when I finally get LangSmith, which helps
inspect what is happening inside chains and agents
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
LangSmith = False
if LangSmith:
    export LANGCHAIN_TRACING_V2="true"
    export LANGCHAIN_API_KEY=""

# Selecting and initializing the desired LLM
gptLLM = OpenAI(model_name = "text-davinci-003")
chatLLM = ChatOpenAI(model_name = "gpt-3.0")



"""
STEP 1: Load the data using DataLoaders. This will generate Documents
"""
# Load in CSV data using a CSV Loader
from langchain.document_loaders.csv_loader import CSVLoader
csv_loader = CSVLoader(file_path='./example_data/mlb_teams_2012.csv')
csv_data = csv_loader.load()

# Load files in a directory
from langchain.document_loaders import TextLoader
directory_loader = DirectoryLoader('../', glob="**/*.md", loader_cls=TextLoader, silent_errors = True)
    # silent_errors = True skips the files which can't be loaded without generating errors
directory_data = directory_loader.load()

# Load Python source code
from langchain.document_loaders import PythonLoader
text_loader_kwargs={'autodetect_encoding': True}
python_loader = DirectoryLoader('../../../../../', glob="**/*.py", loader_cls=PythonLoader, loader_kwargs=text_loader_kwargs)
    # The autodetect_encoding allows the TextLoader to auto detect the encodings of the files before generating errors
python_data = python_loader.load()

# Load HTML documents
from langchain.document_loaders import UnstructuredHTMLLoader
html_loader = UnstructuredHTMLLoader("example_data/fake-content.html")
html_data = html_loader.load()

# Load HTML with BeautifulSoup4
from langchain.document_loaders import BSHTMLLoader
html_bs4_loader = BSHTMLLoader("example_data/fake-content.html")
html_bs4_data = html_bs4_loader.load()

# Load Markdown files
from langchain.document_loaders import UnstructuredMarkdownLoader
md_path = "../../../../../README.md"
md_loader = UnstructuredMarkdownLoader(md_path)
md_data = md_loader.load()

# Load PDF files
from langchain.document_loaders import PyPDFLoader
pdf_loader = PyPDFLoader("example_data/layout-parser-paper.pdf")
pdf_pages = pdf_loader.load_and_split()
    # pdf_pages is an array of documents, where each document contains the page content and metadata with page number



"""
STEP 2: Split the large Documents into smaller chunks using Text Splitters
"""
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500, chunk_overlap=100, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)
    # For the variable docs, insert instead the variable name for the Documents
    # all_splits is an array containing the page_content and metadeta for each split



"""
STEP 3: Store the splitted data using Embeddings and Vector Stores
"""
# Using Chroma Vectorstore and OpenAIEmbeddings model
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
chroma_vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())

# Using Ollama Embeddings
from langchain.embeddings import OllamaEmbeddings
ollama_embeddings = OllamaEmbeddings()
query_result = ollama_embeddings.embed_query(text)
    # text can be either an individual text, or a list of texts

# Using OpenAI Embeddings
from langchain.embeddings import OpenAIEmbeddings
open_ai_embeddings = OpenAIEmbeddings()
query_result = open_ai_embeddings.embed_query(text)
    # text can be either an individual text, or a list of texts

# Using Facebook AI Similarity Search (FAISS)
# Don't forget to install the FAISS library with "pip install faiss-cpu"
from langchain.vectorstores import FAISS
faiss_vectorstore = FAISS.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())



"""
STEP 4: Given a user input, retrieve the relevant splits from storage using a Splitter
"""
inputQuestion = "Some random question"

# For FAISS Vector Store, but very similar for other Vector Stores
documents = faiss_vectorstore.similarity_search(inputQuestion)
    # documents is an array of documents, where each document contains the page content and metadata with page number

# For asynchronous operations (STEPS 3 and 4 together)
# Don't forget to install Qdrant using "pip install qdrant-client"
from langchain.vectorstores import Qdrant
qdrant_vectorstore = await Qdrant.afrom_documents(documents, embeddings, "http://localhost:6333")
documents = await qdrant_vectorstore.asimilarity_search(inputQuestion)

# Using a MultiQueryRetriever, which generates variants of the input question to improve retrieval hit rates
from langchain.retrievers.multi_query import MultiQueryRetriever
chat_llm = ChatOpenAI(temperature=0)
retriever_from_llm = MultiQueryRetriever.from_llm(
    retriever=chroma_vectorstore.as_retriever(), llm=chat_llm
)
# Set logging for the queries. I doubt these next 3 lines are necessary
import logging
logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)
documents = retriever_from_llm.get_relevant_documents(query=inputQuestion)



"""
STEP 5: A ChatModel/ LLM generates a response using prompts
"""
from langchain.chat_models import ChatOpenAI
from langchain import hub

chat_llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# Loading a prompt from LangChain prompt hub
prompt = hub.pull("rlm/rag-prompt")
print(
    prompt.invoke(
        {"context": "filler context", "question": inputQuestion}
    ).to_string()
)

# Useful for debugging with LangSmith
if LangSmith:
    from langchain.schema import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    for chunk in rag_chain.stream("What is Task Decomposition?"):
        print(chunk, end="", flush=True)