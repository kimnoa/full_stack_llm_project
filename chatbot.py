import os, getpass, pprint

from langchain_chroma import Chroma
from langchain_upstage import UpstageEmbeddings, ChatUpstage
from langchain.docstore.document import Document
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.chat import MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage

import streamlit as st

solar_api_key=os.environ["UPSTAGE_API_KEY"]

# Load chromaDB 
vectorstore = Chroma(
    collection_name="sogang_vectorDB",
    embedding_function=UpstageEmbeddings(model="solar-embedding-1-large"),
    persist_directory="./DB_sample" #file 위치 확인
    )

# More general chat
def chain(history:list, human_input:str)->str:
    rag_with_history_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
    You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question considering the history of the conversation. 
    If you don't know the answer, just say that you don't know. 
    ---
    CONTEXT:
    {context}
            """,
            ),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{human_input}"),
        ]
    )

    llm = ChatUpstage()
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={'k':5})
    context = retriever.invoke(human_input)
    print(context)
    chain = rag_with_history_prompt | llm | StrOutputParser()
    query1 = human_input
    response1 = chain.invoke({"history": history, "context": context, "human_input": query1})
    # print(response1)
    return response1

with st.sidebar:
    solar_api_key = st.text_input("Solar API Key", key="solar_api_key", type="password")

st.title("Sogang-i(서강이)")
st.caption("🚀 A Streamlit chatbot powered by SOLAR-mini(Upstage)")
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question considering the history of the conversation. 
If you don't know the answer, just say that you don't know. """}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    if not solar_api_key:
        st.info("Please add your SOLAR API key to continue.")
        st.stop()

    # client = 
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    # response = client.chat.completions.create(model="gpt-3.5-turbo", messages=st.session_state.messages)
    response = chain(st.session_state["messages"],prompt)
    # msg = response.choices[0].message.content
    msg = response
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)