import streamlit as st
from langchain_core.messages import HumanMessage, AIMessageChunk
from langgraph_backend import chatbot

CONFIG = {"configurable": {"thread_id": "thread-1"}}

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

# Display prior message history
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Type here...")

if user_input:
    # Append and display user message
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Define a generator function for st.write_stream
    def generate_response():
        for message_chunk, metadata in chatbot.stream(
            {"messages": [HumanMessage(content=user_input)]},
            config=CONFIG,
            stream_mode="messages",
        ):
            # Ensure we only stream content from AIMessage chunks and avoid tool call chunks
            if isinstance(message_chunk, AIMessageChunk) and message_chunk.content:
                # Handle cases where content might be a list (multimodal/structured) or a string
                if isinstance(message_chunk.content, list):
                    for block in message_chunk.content:
                        if isinstance(block, dict) and "text" in block:
                            yield block["text"]
                        elif isinstance(block, str):
                            yield block
                else:
                    yield message_chunk.content

    # Stream the assistant's response to UI and capture the final full string
    with st.chat_message("assistant"):
        ai_message = st.write_stream(generate_response())

    # Save assistant response to history
    if ai_message:
        st.session_state["message_history"].append(
            {"role": "assistant", "content": ai_message}
        )
