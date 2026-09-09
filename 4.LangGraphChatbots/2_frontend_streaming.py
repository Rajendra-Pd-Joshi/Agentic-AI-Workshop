import streamlit as st
from backend import chatbot
from langchain_core.messages import HumanMessage,AIMessage
user_input = st.chat_input('Type here')

if 'message_history' not in st.session_state:
    st.session_state['message_history']=[]

# load the messages from the st.session_state['message_history']
for message in st.session_state['message_history']:
    with st.chat_message(name=message['role']):
        st.text(message['content'])


if user_input:

    st.session_state['message_history'].append({'role':'user','content':user_input})
    with st.chat_message(name='user'):
        st.text(user_input)

# here chatbot.stream give tuple(message_chunk,metadata) read the documentation of langgraph
    generator = chatbot.stream(input={'messages':[HumanMessage(content=user_input)]},
                               config={'configurable':{'thread_id':1}},
                               stream_mode='messages')
                
# generator function
    def message_generator():
        for message_chunk,metadata in generator:
            if message_chunk.content:
                yield message_chunk.content

    with st.chat_message(name='assistant'):
        ai_message = st.write_stream(stream=message_generator())
        st.session_state['message_history'].append({'role':'assistant','content': ai_message})

    # similar alternative for this  
    # with st.chat_message(name='assistant'):
    #     ai_message = st.write_stream(message_chunk.content for message_chunk,metadata in generator if message_chunk.content)
    #     st.session_state['message_history'].append({'role':'assistant','content':ai_message})