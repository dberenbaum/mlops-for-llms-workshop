"""Python file to serve as the frontend"""
import json
import streamlit as st
from streamlit_chat import message

from qa import get_retriever, get_llm, get_prompt, chain

retriever = get_retriever()
llm = get_llm()
prompt = get_prompt()

# From here down is all the StreamLit UI.
st.set_page_config(page_title="Git QA Bot", page_icon=":robot:")
st.header("Git QA Bot")

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []


def get_text():
    input_text = st.text_input("You: ", "", key="input")
    return input_text


user_input = get_text()

if user_input:
    _, result = chain(user_input, retriever, llm, prompt)

    log_data = {'user_input': user_input, 'answer': result}
    log_str = json.dumps(log_data)
    assert len(log_str.splitlines()) == 1
    with open('data/chat.log', 'a') as f:
        f.write(log_str)
        f.write('\n')

    st.session_state.past.append(user_input)
    st.session_state.generated.append(result)

if st.session_state["generated"]:

    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
