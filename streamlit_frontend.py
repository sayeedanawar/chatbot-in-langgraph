import streamlit as st
from langchain_core.messages import HumanMessage
from langgraph_backend import chatbot

CONFIG = {"configurable": {"thread_id": "1"}}


if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

# message_history = []

for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.text(message["content"])


user_input = st.chat_input("type here")

if user_input:
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.text(user_input)

    response = chatbot.invoke(
        {"messages": [HumanMessage(content=user_input)]}, config=CONFIG
    )
    ai_message = response["messages"][-1].content[0]["text"]
    print(ai_message)

    st.session_state["message_history"].append(
        {"role": "assitant", "content": ai_message}
    )
    with st.chat_message("assitant"):
        st.text(ai_message)
