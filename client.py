import requests
import streamlit as st

st.title("FastAPI con Chat")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt:= st.chat_input("Escribe tu prompt en este espacio"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.text(prompt)
    
    response = requests.get(
        f"http://localhost:8000/generate/text", params={"prompt": prompt}
    )

    response.raise_for_status()

    with st.chat_message("assistant"):
        st.markdown(response.text)