OPENAI_API_KEY = ""
from langchain.embeddings import OpenAIEmbeddings

class WiseSage(object):

    """
    The steps for using Langchain on private, external information are as follows:
    1) Load the data using DataLoaders. This will generate Documents
    2) Split the large Documents into smaller chunks using Text Splitters
    3) Store the splitted data using Embeddings and Vector Stores
    4) Given a user input, retrieve the relevant splits from storage using a Splitter
    5) A ChatModel/ LLM generates a response using prompts
    """
    # pip install langchain
    # pip install openai
    # pip install chromadb
    # pip install tiktoken
    # pip install unstructured
    # pip install pypdf

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
        length = len(filepath)
        filepath_backwards = filepath[::-1]
        index = filepath_backwards.find('.')
        file_format = filepath[length-index:length]
        if file_format == "csv":
            print("Detected .csv")
            # Load in CSV data using a CSV Loader
            from langchain.document_loaders.csv_loader import CSVLoader
            csv_loader = CSVLoader(file_path=filepath)
            csv_data = csv_loader.load()
            return csv_data
        elif file_format == "py":
            print("Detected .py")
            # Load Python source code
            from langchain.document_loaders import PythonLoader
            python_loader = PythonLoader(filepath)
            python_data = python_loader.load()
            return python_data
        elif file_format == "html":
            print("Detected .html")
            # Load HTML documents
            from langchain.document_loaders import UnstructuredHTMLLoader
            html_loader = UnstructuredHTMLLoader(filepath)
            html_data = html_loader.load()
            return html_data
        elif file_format == "pdf":
            print("Detected .pdf")
            # Load PDF files
            from langchain.document_loaders import PyPDFLoader
            pdf_loader = PyPDFLoader(filepath)
            pdf_pages = pdf_loader.load_and_split()
                # pdf_pages is an array of documents, where each document contains the page content and metadata with page number
            return pdf_pages
        elif file_format == "txt":
            print("Detected .txt")
             # Load files in a directory
            from langchain.document_loaders import TextLoader
            text_loader = TextLoader(filepath)
            text_data = text_loader.load()
            return text_data            
        else:
            print("Detected something unfamiliar")
            # Load all .txt files in a Directory
            from langchain.document_loaders import TextLoader
            from langchain.document_loaders import DirectoryLoader
            directory_loader = DirectoryLoader(filepath, glob="**/*.txt", loader_cls=TextLoader, silent_errors = True)
                # silent_errors = True skips the files which can't be loaded without generating errors
            directory_data = directory_loader.load()
            return directory_data
        """
        # Load HTML with BeautifulSoup4
        from langchain.document_loaders import BSHTMLLoader
        html_bs4_loader = BSHTMLLoader(filepath)
        html_bs4_data = html_bs4_loader.load()
        return html_bs4_data
        """



    """
    STEP 2: Split the large Documents into smaller chunks using Text Splitters
    """
    def SplitData(self, documents):
        from langchain.text_splitter import RecursiveCharacterTextSplitter

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300, chunk_overlap=60, add_start_index=True
        )
        all_splits = text_splitter.split_documents(documents)
            # For the variable documents, insert instead the variable name for the Documents
            # all_splits is an array containing the page_content and metadeta for each split
        return all_splits



    """
    The following decorator and methods are included to prevent this program from hitting the rate limits
    """
    # imports
    import random
    import time

    import openai
    from openai import OpenAI
    client = OpenAI(api_key = OPENAI_API_KEY)

    # define a retry decorator
    def retry_with_exponential_backoff(
        func,
        initial_delay: float = 1,
        exponential_base: float = 2,
        jitter: bool = True,
        max_retries: int = 20,
        errors: tuple = (openai.RateLimitError,),
    ):
        """Retry a function with exponential backoff."""

        def wrapper(*args, **kwargs):
            # Initialize variables
            num_retries = 0
            delay = initial_delay

            # Loop until a successful response or max_retries is hit or an exception is raised
            while True:
                try:
                    return func(*args, **kwargs)

                # Retry on specified errors
                except errors as e:
                    # Increment retries
                    num_retries += 1

                    # Check if max retries has been reached
                    if num_retries > max_retries:
                        raise Exception(
                            f"Maximum number of retries ({max_retries}) exceeded."
                        )

                    # Increment the delay
                    delay *= exponential_base * (1 + jitter * random.random())

                    # Sleep for the delay
                    time.sleep(delay)

                # Raise exceptions for any errors not specified
                except Exception as e:
                    raise e

        return wrapper



    """
    STEP 3: Save to disk the splitted data using Embeddings and Vector Stores
    """
    @retry_with_exponential_backoff
    def StoreData(self, all_splits):

        # Using Chroma Vectorstore and OpenAIEmbeddings model
        from langchain.vectorstores import Chroma
        from langchain.embeddings import OpenAIEmbeddings
        embed = OpenAIEmbeddings(openai_api_key = OPENAI_API_KEY)
        vectorstore = Chroma.from_documents(documents=all_splits, embedding=embed, persist_directory = "chroma_vectorstore")
        # cleanup
        # vectorstore.delete_collection()
        # return embed, vectorstore
        
        
        
        """
        from langchain.indexes import SQLRecordManager, index
        collection_name = "religious_works"
        chroma_vectorstore = Chroma(es_url = "http://localhost:9200", index_name = collection_name, embedding=OpenAIEmbeddings(openai_api_key = OPENAI_API_KEY))
        namespace = f"Chroma/{collection_name}"
        record_manager = SQLRecordManager(
            namespace, db_url="sqlite:///record_manager_cache.sql"
        )
        record_manager.create_schema()

        # Hacky helper method to clear content.
        # We essentially index into the vectorstore, but with an empty list of split documents
        def _clear():
            index([], record_manager, chroma_vectorstore, cleanup="full", source_id_key="source")
        _clear()
        # Indexing all the documents into the record_manager
        # If cleanup="full" or "indexed", when a source file is modified, all Documents associated with that
            # source file are deleted and replaced with newer versions. In "full", when a split Document isn't listed
            # in the first parameter in index(), that Document is deleted from the vectorstore; with "indexed", the
            # Document doesn't immediately get deleted
        index(all_splits, record_manager, chroma_vectorstore, cleanup="full", source_id_key="source")
        """



    """
    STEP 4: Given a user input, retrieve the relevant splits from storage using a Splitter
    """
    @retry_with_exponential_backoff
    def RetrieveData(self, inputQuestion, embedding):
        from langchain.vectorstores import Chroma
        # For any Vector Store
        retriever = Chroma(persist_directory ="chroma_vectorstore", embedding_function = embedding).as_retriever(search="similarity_search")
        # retriever is VectorStoreRetriever object. Its relevant splits can be extracted with the line of code below
        # relevant_splits = retriever.get_relevant_documents(inputQuestion)
        # print("The page content in the first relevant split is ", relevant_splits[0].page_content)
        # print("The number of relevant splits in the retriver is ", len(relevant_splits))
        return retriever
        """
        relevantSplits = vectorstore.similarity_search(inputQuestion)
            # relevantSplits is an array of documents, where each document contains the page content and metadata with page number
        return relevantSplits
        """

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
    @retry_with_exponential_backoff
    def GenerateResponse(self, inputQuestion, retriever):
        from langchain.chat_models import ChatOpenAI
        from langchain import hub

        chat_llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key = OPENAI_API_KEY)

        
        # Creating a custom prompt for memory incorporation
        from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
        from langchain.schema import StrOutputParser
        from langchain_core.runnables import RunnablePassthrough

        
        condense_q_system_prompt = "Given a chat history and the latest user question \
            which might reference the chat history, formulate a standalone question \
            which can be understood without the chat history. Do NOT answer the question, \
            just reformulate it if needed and otherwise return it as is."
        # MessagesPlaceholder will be replaced by the content in the variable in parenthesis once invoked.
        condense_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", condense_q_system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", inputQuestion),
            ]
        )
        condense_q_chain = condense_q_prompt | chat_llm | StrOutputParser()

        # The following is the prompt for the first message in the chat, when there's no chat history
        qa_system_prompt = "You are an assistant for question-answering tasks. \
            Use the following pieces of retrieved context to answer the question. \
            If you don't know the answer, just say that you don't know. \
            Keep the answer concise. \
            End all responses with I have spoken \
            Context: {context} \
            Answer:"
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", qa_system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", inputQuestion),
            ]
        )

        # The following decides which question should be used depending on if there's a chat history or not
        def condense_question(input: dict):
            if not (input["chat_history"] == []):
                # print("input[\"chat_history\"] is ", input["chat_history"])
                condensed_question = condense_q_chain.invoke(
                    {
                        "chat_history": self.chat_history, "question": inputQuestion
                    }
                )
                # print("Inside condense_question, the condensed question is ", condensed_question)
                return condensed_question
            else:
                return input["question"]

        # Now the LCEL Runnable protocol which defines the chain
        # RunnableLambda is like a lambda expression in Java.
        # RunnableLambda(lambda x: x+1) or just lambda x: x+1 is a function which takes in x and returns x+1
        # The chain is a langchain_core.runnables.base.Runnable, which likely is either a RunnableSequence or RunnableParallel
        # RunnableSequence invokes a series of runnables sequentially using the | operator or passing a list of runnables to RunnableSequence
        # RunnableParallel invokes runnables concurrently using a dict literal within a sequence or passing a dict with runnable value entries to RunnableParallel
        # doing runnable.invoke() will generate a RunnableSequence while runnable.batch([]) will generate a RunnableParallel
        # Below, rag_chain is actually a RunnableSequence, though it might seem like a RunnableParallel
        # RunnablePassthrough() allows inputs to pass through unchanged. It's also capable of adding new keys to the output if the input is a dict
        #   To add new keys, use RunnablePassthrouh.assign(new_key = new_value)
        def format_docs(retriever_input):
            return "\n\n".join(doc.page_content for doc in retriever_input)    
        
        rag_chain = (
            RunnablePassthrough.assign(context=condense_question | retriever | format_docs)
            | qa_prompt
            | chat_llm
        )

        # Updating the chat history
        from langchain_core.messages import HumanMessage
        # for chunk in rag_chain.stream(inputQuestion):
            # print(chunk, end="", flush=True)
        ai_msg = rag_chain.invoke({"question": inputQuestion, "chat_history": self.chat_history}) # type is AIMessage
        self.chat_history.extend([HumanMessage(content=inputQuestion), ai_msg])
        print(ai_msg)
        # print("The rag_chain is ", rag_chain)



    def main(self):
        self.chat_history = []
        embedding = OpenAIEmbeddings(openai_api_key = OPENAI_API_KEY)
        while True:
            procedure = input("Enter 1 for creating a vectorstore. Enter 2 for asking a question.")
            if procedure == "1":
                filepaths = input("Enter the name(s) of the file(s) you want to be stored in a vectorestore.\n Supported file types are .csv, .py, .html, .pdf, and .txt")
                filepaths = filepaths.split()
                documents = []
                for filepath in filepaths:
                    documents.extend(self.LoadData(filepath))
                split_data = self.SplitData(documents)
                self.StoreData(split_data)
                print("Done generating the vectorstore.")
            elif procedure == "2":
                inputQuestion = input("Please type your question and click Enter.")
                retriever = self.RetrieveData(inputQuestion, embedding)
                self.GenerateResponse(inputQuestion, retriever)
            else:
                break
        print("Thank you for using this chatbot! Have a good day!")




if __name__ == "__main__":
    # print("In the main method")
    beginner_learner = WiseSage()
    beginner_learner.main()
