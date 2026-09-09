import streamlit as st
from backend import chatbot
from langchain_core.messages import HumanMessage

user_input = st.chat_input('type here...')

if 'message_history' not in st.session_state:
    st.session_state['message_history']=[]

# load the messages from the st.session_state['message_history]
for message in st.session_state['message_history']:
    with st.chat_message(name = message['role']):
        st.text(message['content'])


if user_input:
    st.session_state['message_history'].append({'role':'human','content':user_input})
    with st.chat_message(name = 'human'):
        st.text(user_input)

    response = chatbot.invoke(input = {'messages':[HumanMessage(content=user_input)]},config = {'configurable':{'thread_id':'session-1'}})
    ai_message = response['messages'][-1].content

    st.session_state['message_history'].append({'role':'ai','content':ai_message})
    with st.chat_message(name = 'ai'):
        st.text(ai_message)
    