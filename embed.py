from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_community.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

class Chatbot:
    def __init__(self):

    #embedding model
        self.embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # Load the website info
        with open("D:/langchain/eco_cb/ecobliss.txt", "r", encoding="utf-8") as f:
            website_text = f.read()

        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_text(website_text)

        # Create Document objects
        documents = [Document(page_content=chunk) for chunk in chunks]

        # Create FAISS vector store
        self.vectorstore = FAISS.from_documents(documents, self.embedding_model)
        self.retriever = self.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

        # def find_best_chunks(self, query, top_k=3):
        #     results = self.vectorstore.similarity_search(query, k=top_k)
        #     return [doc.page_content for doc in results]



        # Setup chatbot
        self.llm = ChatOpenAI(model="meta-llama/llama-3.1-8b-instruct:free")



        # Initialize system message
        self.system_prompt = """
        You are a customer support agent for EcoBliss - plant ecommerce website. You are also plant specialist. 
        You are very knowledgeable about plants and their care. You are friendly and helpful.
        You are a great assistant for anyone who needs help with plants.
        You are limited in your knowledge about anything other than plants, our website, and customer service.
        If you are asked anything that does not relate to the above mentioned, politely refuse to answer.

        Answer user questions politely and only using the information provided. If the question is outside the scope, politely refuse.

        Don't do too much cross-questioning. 
        Don't suggest plants yourself as stocks of plants may vary and we may not have a plant you suggest.
        Don't tell the price of any plant.
        Don't give offers and discounts on your own. 
        Use only the information given in the provided text and don't hallucinate anything yourself.
        You are not an AI agent you cannot do anything on behalf of the customer.

        Relevant information from our website:
        {context}
        """

        self.chat_history = []


    def ask(self, user_input: str) -> str:
        docs = self.retriever.invoke(user_input)
        context = "\n\n".join([doc.page_content for doc in docs])

        current_system_message = SystemMessage(content=self.system_prompt.format(context=context))

        messages = [current_system_message] + self.chat_history + [HumanMessage(content=user_input)]

        result = self.llm.invoke(messages)

        self.chat_history.append(HumanMessage(content=user_input))
        self.chat_history.append(AIMessage(content=result.content))
        if len(self.chat_history) > 10:
            self.chat_history = self.chat_history[-10:]

        return result.content
    
if __name__ == "__main__":
    cb = Chatbot()
    while True:
        user_msg = input("You: ")
        if user_msg.lower() == "exit":
            break
        response = cb.ask(user_msg)
        print("Bot:", response)