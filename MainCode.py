OPENAI_API_KEY = "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"

class WiseSage(object):

    """
    The steps for using Langchain on private, external information are as follows:
    1) Load the data using DataLoaders. This will generate Documents
    2) Split the large Documents into smaller chunks using Text Splitters
    3) Store the splitted data using Embeddings and Vector Stores
    4) Given a user input, retrieve the relevant splits from storage using a Splitter
    5) A ChatModel/ LLM generates a response using prompts
    """

    # Imports
    # from dotenv import load_dotenv, find_dotenv

    # load_dotenv(find_dotenv())

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    The below lines are for when I finally get LangSmith, which helps
    inspect what is happening inside chains and agents
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    LangSmith = False
    if LangSmith:
        LANGCHAIN_TRACING_V2="true"
        LANGCHAIN_API_KEY=""

    """
    STEP 1: Load the data using DataLoaders. This will generate Documents
    """
    def LoadData(self, filepath):
        """
        # Load in CSV data using a CSV Loader
        from langchain.document_loaders.csv_loader import CSVLoader
        csv_loader = CSVLoader(file_path=filepath)
        csv_data = csv_loader.load()
        return csv_data

        # Load files in a directory
        from langchain.document_loaders import TextLoader
        directory_loader = DirectoryLoader(filepath, glob="**/*.md", loader_cls=TextLoader, silent_errors = True)
            # silent_errors = True skips the files which can't be loaded without generating errors
        directory_data = directory_loader.load()
        return directory_data

        # Load Python source code
        from langchain.document_loaders import PythonLoader
        text_loader_kwargs={'autodetect_encoding': True}
        python_loader = DirectoryLoader(filepath, glob="**/*.py", loader_cls=PythonLoader, loader_kwargs=text_loader_kwargs)
            # The autodetect_encoding allows the TextLoader to auto detect the encodings of the files before generating errors
        python_data = python_loader.load()
        return python_data

        # Load HTML documents
        from langchain.document_loaders import UnstructuredHTMLLoader
        html_loader = UnstructuredHTMLLoader(filepath)
        html_data = html_loader.load()
        return html_data

        # Load HTML with BeautifulSoup4
        from langchain.document_loaders import BSHTMLLoader
        html_bs4_loader = BSHTMLLoader(filepath)
        html_bs4_data = html_bs4_loader.load()
        return html_bs4_data

        # Load Markdown files
        from langchain.document_loaders import UnstructuredMarkdownLoader
        md_path = filepath
        md_loader = UnstructuredMarkdownLoader(md_path)
        md_data = md_loader.load()
        return md_data
        """

        # Load PDF files
        from langchain.document_loaders import PyPDFLoader
        pdf_loader = PyPDFLoader(filepath)
        pdf_pages = pdf_loader.load_and_split()
            # pdf_pages is an array of documents, where each document contains the page content and metadata with page number
        return pdf_pages



    """
    STEP 2: Split the large Documents into smaller chunks using Text Splitters
    """
    def SplitData(self, documents):
        from langchain.text_splitter import RecursiveCharacterTextSplitter

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=20, chunk_overlap=5, add_start_index=True
        )
        all_splits = text_splitter.split_documents(documents)
            # For the variable documents, insert instead the variable name for the Documents
            # all_splits is an array containing the page_content and metadeta for each split
        return all_splits



    """
    STEP 3: Store the splitted data using Embeddings and Vector Stores
    """
    def StoreData(self, all_splits, text):
        # Using Chroma Vectorstore and OpenAIEmbeddings model
        from langchain.vectorstores import Chroma
        from langchain.embeddings import OpenAIEmbeddings
        chroma_vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings(openai_api_key = OPENAI_API_KEY))
        return chroma_vectorstore

        """
        # Using Ollama Embeddings
        from langchain.embeddings import OllamaEmbeddings
        ollama_embeddings = OllamaEmbeddings()
        query_result = ollama_embeddings.embed_query(text)
            # text can be either an individual text, or a list of texts
        return query_result

        # Using OpenAI Embeddings
        from langchain.embeddings import OpenAIEmbeddings
        open_ai_embeddings = OpenAIEmbeddings()
        query_result = open_ai_embeddings.embed_query(text)
            # text can be either an individual text, or a list of texts
        return query_result

        # Using Facebook AI Similarity Search (FAISS)
        # Don't forget to install the FAISS library with "pip install faiss-cpu"
        from langchain.vectorstores import FAISS
        faiss_vectorstore = FAISS.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())
        return faiss_vectorstore
        """



    """
    STEP 4: Given a user input, retrieve the relevant splits from storage using a Splitter
    """
    def RetrieveData(self, inputQuestion, vectorstore):
        # For any Vector Store
        relevantSplits = vectorstore.similarity_search(inputQuestion)
            # relevantSplits is an array of documents, where each document contains the page content and metadata with page number
        return relevantSplits

        """
        # For asynchronous operations (STEPS 3 and 4 together)
        # Don't forget to install Qdrant using "pip install qdrant-client"
        from langchain.vectorstores import Qdrant
        qdrant_vectorstore = await Qdrant.afrom_documents(documents, embeddings, "http://localhost:6333")
        relevantSplits = await qdrant_vectorstore.asimilarity_search(inputQuestion)
        return relevantSplits

        # Using a MultiQueryRetriever, which generates variants of the input question to improve retrieval hit rates
        from langchain.retrievers.multi_query import MultiQueryRetriever
        chat_llm = ChatOpenAI(temperature=0)
        retriever_from_llm = MultiQueryRetriever.from_llm(
            retriever=vectorstore.as_retriever(), llm=chat_llm
        )
        # Set logging for the queries. I doubt these next 3 lines are necessary
        import logging
        logging.basicConfig()
        logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)
        relevantSplits = retriever_from_llm.get_relevant_documents(query=inputQuestion)
        return relevantSplits
        """



    """
    STEP 5: A ChatModel/ LLM generates a response using prompts
    """
    def GenerateResponse(self, inputQuestion, documents, relevantSplits):
        from langchain.chat_models import ChatOpenAI
        from langchain import hub

        chat_llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

        # Loading a prompt from LangChain prompt hub
        prompt = hub.pull("rlm/rag-prompt")
        prompt.invoke(
            {"context": "filler context", "question": inputQuestion}
        ).to_string()

        # LCEL Runnable protocol to define the chain
        from langchain.schema import StrOutputParser
        from langchain_core.runnables import RunnablePassthrough

        def format_docs(documents):
            return "\n\n".join(doc.page_content for doc in documents)

        rag_chain = (
            {"context": relevantSplits | format_docs, "question": RunnablePassthrough()}
            | prompt
            | chat_llm
            | StrOutputParser()
        )
        for chunk in rag_chain.stream(inputQuestion):
            print(chunk, end="", flush=True)


    def main(self):
        filepath = "testData.pdf"
        someText = "Test. Abdulrahman is faster than Ameer"
        inputQuestion = input("Please type your question and click Enter.")
        documents = self.LoadData(filepath)
        split_data = self.SplitData(documents)
        vectorstore = self.StoreData(split_data, someText)
        relevantSplits = self.RetrieveData(inputQuestion, vectorstore)
        self.GenerateResponse(inputQuestion, documents, relevantSplits)




if __name__ == "__main__":
    print("In the main method")
    beginner_learner = WiseSage()
    beginner_learner.main()
