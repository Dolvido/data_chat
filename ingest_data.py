import os
from dotenv import load_dotenv, find_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import UnstructuredAPIFileLoader
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub
from langchain.memory import ConversationBufferMemory
import pickle

class EnvironmentSetup:
    def __init__(self):
        self.load_environment_variables()
    
    def load_environment_variables(self):
        load_dotenv(find_dotenv())  # Uncomment in your environment
        HUGGINGFACEHUB_API_TOKEN = os.environ["HUGGINGFACEHUB_API_TOKEN"]  # Uncomment in your environment

        print("Environment variables loaded.")


class DataIngestor:
    def __init__(self, directory_path='conv_data'):
        self.directory_path = directory_path
        self.raw_docs = []
        self.documents = []
        self.load_and_process_data()

    def load_and_process_data(self):
        # Uncomment these lines in your environment
        filenames = [os.path.join(self.directory_path, filename) for filename in os.listdir(self.directory_path)]
        loader = UnstructuredAPIFileLoader(file_path=filenames, api_key="ad3ZymidTRI5E5aR89xLptMKCwwjZO")
        self.raw_docs = loader.load()
        text_splitter = CharacterTextSplitter(separator="\n\n", chunk_size=600, chunk_overlap=100, length_function=len)
        self.documents = text_splitter.split_documents(self.raw_docs)
        print("Data loaded and processed.")


class MemoryStore:
    def __init__(self, documents):
        self.documents = documents
        self.vectorstore = None
        self.create_vector_store()

    def create_vector_store(self):
        embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
        self.vectorstore = FAISS.from_documents(self.documents, embeddings)
        print("Vectorstore created.")


class ConversationalChain:
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore
        self.conversation_chain = None
        self.create_conversational_chain()

    def create_conversational_chain(self):
        llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.1, "max_length":512})
        memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
        self.conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=self.vectorstore.as_retriever(), memory=memory)
        print("Conversational chain created.")

    def query_last_conversation(self, query):
        response = self.conversation_chain({'question': query})
        chat_history = response['chat_history']
        # chat_history = []  # Placeholder, uncomment the above lines in your environment

        for i, message in enumerate(chat_history):
            if i % 2 == 0:
                print("Question: " + message.content)
            else:
                print(message.content)


# Testing the encapsulated functionalities (without actual data processing due to environment constraints)
env_setup = EnvironmentSetup()
data_ingestor = DataIngestor()
memory_store = MemoryStore(data_ingestor.documents)
conversational_chain = ConversationalChain(memory_store.vectorstore)
conversational_chain.query_last_conversation("What was the topic?")
conversational_chain.query_last_conversation("What was the tone?")
conversational_chain.query_last_conversation("Who proposed what?")
conversational_chain.query_last_conversation("What was the outcome?")
conversational_chain.query_last_conversation("How many turns of conversation were there?")
conversational_chain.query_last_conversation("Which LLM_Chain contributed more to the conversation?")



