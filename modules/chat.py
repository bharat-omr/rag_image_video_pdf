from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from modules.ai_services import get_chat_model

def get_conversation_chain(vectorstore):
    llm = get_chat_model()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=vectorstore.as_retriever(), memory=memory
    )
