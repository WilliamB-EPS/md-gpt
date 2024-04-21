"""
William Bowers
EPS Adv. Prog: TiCS (AIML)
March 6 2024

A streamlit web app for a biomedical chatbot.

IMPORTANT: to run the chatbot (build the app), use the command:

    streamlit run app.py

IMPORTANT: use the knowledge_embedding python notebook to create the my_vector_db
directory and store the vector db. this is REQUIRED to run this app.

NOTE: for the actual web app, some code has been used from the streamlit docs:
documentation:

https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps

The code has been altered somewhat to fit this project. The code used is not 
relevant to the learning goals of this project since it deals with concepts
specific to streamlit web apps, and not machine learning. I was also given
permission in class to do this.
"""

import streamlit as st
import time
from langchain.chat_models import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

################################# SETUP #####################################

# setup function that we run only ONCE in our web app. streamlit will 
# essentially rerun this file every time the user enters a prompt. therefore, 
# we want to only run our setup code when we have to. this saves a lot of time
def setup():

    # define the embedding function we use for our text. using a FREE hugging 
    # face model instead of openai, which costs money
    embedding_func = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # we start by obtaining our vector database using chroma. this will allow
    # us to access text that is similar to the user prompt
    # NOTE: make sure this matches the chroma directory with the correct vector
    # database (if you have multiple for some reason)
    vector_db = Chroma(
        persist_directory="./my_vector_db", 
        embedding_function=embedding_func
    )

    # setup the LLM using ChatGPT. using Mr. Briggs' open api key for this.
    llm = ChatOpenAI()

    # prompt that we feed the LLM, including the context. this is the center
    # of knowledge embedding
    prompt_skeleton = """
    Imagine you are a doctor helping patients understanding medical conditions.
    You are being sent a question from a client and need to respond. Together 
    with their question, you will also recieve some context on that topic. 
    DO NOT mention that you have recieved this context. When responding, 
    incorporate as much of the context as possible. Make sure your response is 
    specific and detailed but not longer than 300 words. If the query doesn't 
    match the context provided, do the best with whatever knowledge you have. 

    Here is your client's question: {query}

    Here is the context that I have fetched for you: {context}
    """

    # store everything we will need later as session states. this means we can
    # access everything easily without re-doing any computations, since session
    # states are stored for the entire session (obviously).
    st.session_state.vector_db = vector_db
    st.session_state.model = llm
    st.session_state.prompt_skeleton = prompt_skeleton

# if we havent stored what we need, that means we haven't set up. do that now
if "chain_model" not in st.session_state:
    setup()

# function that takes the query and gives the response
def get_model_response(user_query):

    # run the similarity search on the vector db. get 3 relevant text snippets
    relevant_info = st.session_state["vector_db"].similarity_search(
                                                            user_query, 
                                                            k=3)

    # fetch the actual results from the search (drop the metadata and other
    # unnecessary stuff)
    context = [item.page_content for item in relevant_info]

    # add the query and context into our prompt skeleton
    full_prompt = st.session_state["prompt_skeleton"].format(
                                                            query=user_query, 
                                                            context=context)

    # now input this, along with the user query, into our LLM
    response = st.session_state["model"].invoke(full_prompt)

    # extract just the string (drop unnecessary stuff) and return
    ret_val = response.content
    return str(ret_val)

################################# WEB APP #####################################

# function that builds a streamlit web app which we will use to interface with
# the model.
def build_app():

    # function that returns one word at a time from our response with a small
    # time interval in between. this makes it seem like the model is thinking.
    def response_generator(query):
        answer = get_model_response(query)
        for ind_word in answer.split():
            yield ind_word + " "
            time.sleep(0.05)

    # set our title for this page
    st.title("MD-GPT: LLM for Biomedical Q&A")

    # add a drop down box with a disclaimer that this bot should NOT be used for
    # medical advice and was student developed
    with st.expander("Disclaimer"):
        st.write("This bot does NOT produce professional medical advice.")
        st.write("What you learn  should be checked by a medical professional")
        st.write("This bot is student developed and might make mistakes.")

    # add another drop down box for people curious about how the model works
    with st.expander("How does this work?"):
        st.write("MD-GPT relies on knowledge embedding to produce answers.")
        st.write("Essentially, we have a big database of text related to " 
                 + "biomedicine. Then, we check which part matches your prompt " 
                 + "most closely. We provide this relevant data - along with "
                 + "your prompt - to an LLM.")
        st.write("This lets us answer your questions with added detail!")

    # display an initial message that introduces the model
    init_msg = ("Welcome to MD-GPT! I am specialized in answering questions "
        + "regarding diseases, treatment plans, research, etc. Basically, " 
        + "anything biomedical. I am based off of ChatGTP and further " 
        + "trained from 17,000 medical questions and answers. What would "  
        + "you like to know?")

    # render that introductory message
    with st.chat_message("assistant"):
        st.markdown(init_msg)

    # initialize the chat history with some streamlit stuff. please reference
    # the streamlit documentation for more info about this
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # render each message from the chat history as markdown
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # see if the user has entered a prompt, and, if so, compute a response
    if prompt := st.chat_input("What would you like to know?"):

        # first, log the user's message in the chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # now render their message as markdown
        with st.chat_message("user"):
            st.markdown(prompt)

        # now send the model's message in response
        with st.chat_message("assistant"):
            response = st.write_stream(response_generator(prompt))

        # add the model's response to the history so we track over time
        st.session_state.messages.append({"role": "assistant", "content": response})

# build the app!
build_app()