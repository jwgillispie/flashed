import streamlit as st
from langchain import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader, TextLoader, YoutubeLoader, UnstructuredPDFLoader
from langchain.chains.summarize import load_summarize_chain
from langchain import PromptTemplate
from skllm.preprocessing import GPTSummarizer
from io import BytesIO
from PyPDF2 import PdfReader
import openai
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
import numpy as np
from sklearn.cluster import KMeans

# openai key sk-yZU5QBGPGF7g7D7sDELuT3BlbkFJl8kkYZKX4EGRTbfog1zmj

# Placeholder functions for summarization and flashcards
def summarize_text(text, key):
    # take in the text 
    llm = OpenAI(openai_api_key=key)
    num_tokens = llm.get_num_tokens(text)
    # errors for tokens == 0
    if num_tokens < 4000:
        summary = small_summarizer(text, key)
    elif num_tokens > 4000 and num_tokens <= 90000:
        summary = medium_summarizer(text, key)
    else:
        summary = large_summarizer(text, key)
    return summary


    # tokenn count 

    # pass into differerent functions for each of the token counts 


    ### for each function based on token count ### 
    # create text splitter 

    # create docs with chunks

    # create summarization chain 


    # run summmary chain on docs 

    # return docs 

     # Just an example to take the first 200 chars
def small_summarizer(text, key): # consider allowing user to choose sentence count
    llm = ChatOpenAI(openai_api_key = key)
    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n"], chunk_size=9000, chunk_overlap=500)
    docs = text_splitter.create_documents([text])
    summary_chain = load_summarize_chain(llm=llm,
                                    chain_type='stuff',

    ) 
    summary = summary_chain.run(docs)
    return summary

def medium_summarizer(text, key):
    llm = ChatOpenAI(openai_api_key=key)
    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n"], chunk_size=9000, chunk_overlap=500)
    docs = text_splitter.create_documents([text])

    map_prompt = """
        Write a short informative story about the following texxt:
        "{text}"
        CONCISE SUMMARY:
        """
    map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])
    
    
    combine_prompt = """
        Write a concise summary of the following text delimited by triple backquotes.
        Return your response in at most 13 bullet points which covers the key points of the text.
        ```{text}```
        BULLET POINT SUMMARY:
    """
    combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=["text"])
    
    summary_chain = load_summarize_chain(llm=llm,
                                        chain_type='map_reduce',
                                        map_prompt=map_prompt_template,
                                        combine_prompt=combine_prompt_template,
                                        ) 
    summary = summary_chain.run(docs)
    return summary

def large_summarizer(text, key):
    llm = ChatOpenAI(openai_api_key=key, model="gpt-3.5-turbo")
    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n"], chunk_size=13000, chunk_overlap=0)
    docs = text_splitter.create_documents([text])

    embeddings = OpenAIEmbeddings(openai_api_key=key)

    vectors = embeddings.embed_documents([x.page_content for x in docs])
    num_clusters = 11

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(vectors)
    # Find the closest embeddings to the centroids

# Create an empty list that will hold your closest points
    closest_indices = []

    # Loop through the number of clusters you have
    for i in range(num_clusters):
        
        # Get the list of distances from that particular cluster center
        distances = np.linalg.norm(vectors - kmeans.cluster_centers_[i], axis=1)
        
        # Find the list position of the closest one (using argmin to find the smallest distance)
        closest_index = np.argmin(distances)
        
        # Append that position to your closest indices list
        closest_indices.append(closest_index)
        selected_indices = sorted(closest_indices)


        
    map_prompt = """
        You will be given a single passage of a book. This section will be enclosed in triple backticks (```)
        Your goal is to give a summary of this section so that a reader will have a full understanding of what happened.
        Your response should be at least three paragraphs and fully encompass what was said in the passage.

        ```{text}```
        FULL SUMMARY:
    """
    map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])
    map_chain = load_summarize_chain(llm=llm,
                             chain_type="stuff",
                             prompt=map_prompt_template)
    selected_docs = [docs[doc] for doc in selected_indices]
    # Make an empty list to hold your summaries
    summary_list = []

    # Loop through a range of the lenght of your selected docs
    for i, doc in enumerate(selected_docs):
        
        # Go get a summary of the chunk
        chunk_summary = map_chain.run([doc])
        
        # Append that summary to your list
        summary_list.append(chunk_summary)

    summaries = "\n".join(summary_list)

    # Convert it back to a document
    summaries = Document(page_content=summaries)

    combine_prompt = """
        You will be given a series of summaries from a book. The summaries will be enclosed in triple backticks (```)
        Your goal is to give a verbose summary of what happened in the story using no more than 13 bullet points.
        The reader should be able to grasp what happened in the book.

        ```{text}```
        VERBOSE SUMMARY:
        """
    combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=["text"])

    reduce_chain = load_summarize_chain(llm=llm,
                             chain_type="stuff",
                             prompt=combine_prompt_template,
#                              verbose=True # Set this to true if you want to see the inner workings
                                   )
    summary = reduce_chain.run([summaries])

    return summary


def create_flashcards(text):
    # Replace this with your desired method to create flashcards
    return [text[i:i+50] for i in range(0, len(text), 50)]

def read_youtube(file_path):
    # Replace this with your desired method to read the input file
    loader = YoutubeLoader.from_youtube_url(file_path, add_video_info=True)
    data = loader.load()
    text = ""
    for doc in data:
        text += doc.page_content

    return text


def read_text(file_path):
    loader = TextLoader(file_path)
    data = loader.load()
    text = ""
    for doc in data:
        text += doc.page_content
    return text


st.title("Study Tool")
password = st.text_input('Enter your OpenAI API Key:', type='password')
key = password
option = st.sidebar.selectbox("Choose an option:", ["Home", "Summarize", "Flashcards"])
openai.api_key = password
if option == "Home":
    uploaded_file = st.file_uploader("Choose a file", type=['txt', 'pdf'])
    youtube_url = st.text_input('Enter your YouTube URL', '')

    if uploaded_file:
        if uploaded_file.type == "application/pdf":
            pdf_reader = PdfReader(uploaded_file)

            text = ''
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text(0)


        elif uploaded_file.type == "text/plain":
            loader = TextLoader(uploaded_file)
            data = loader.load()
            text = ""
            for doc in data:
                text += doc.page_content
            text = read_text(uploaded_file)
        # take in a youtube url
        else: 
            loader = YoutubeLoader.from_youtube_url(uploaded_file, add_video_info=True)
            data = loader.load()
            text = ""
            for doc in data:
                text += doc.page_content

        action = st.selectbox("What would you like to do?", ["", "Summarize", "See Flashcards"])

        if action == "Summarize":
            st.session_state['summary'] = summarize_text(text, key)
            st.write("Summary created! Go to the Summarize tab to see it.")
        elif action == "See Flashcards":
            st.session_state['flashcards'] = create_flashcards(text)
            st.write("Flashcards created! Go to the Flashcards tab to see them.")

elif option == "Summarize":
    if 'summary' in st.session_state:
        st.write(st.session_state['summary'])
    else:
        st.write("No summary available. Go back to Home to create one.")

elif option == "Flashcards":
    if 'flashcards' in st.session_state:
        for card in st.session_state['flashcards']:
            st.write(card)
    else:
        st.write("No flashcards available. Go back to Home to create them.")
