import os
import tempfile
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from audio_recorder_streamlit import audio_recorder
import speech_recognition as sr
from io import BytesIO
import base64
from openai import OpenAI # for use in audio response


# setting environment variable
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ["LANGCHAIN_API_KEY"] = 'YOUR-LANGCHAIN-API-KEY'
os.environ['OPENAI_API_KEY']  = 'YOUR-OPENAI-API-KEY'


model = ChatOpenAI(model = 'gpt-4o-mini',temperature = .5)


# Function to process uploaded file
def process_uploaded_file(uploaded_file):
    # Create a temporary file in memory
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        temp_file_path = temp_file.name  # Get the path of the temporary file
    
    # Load the file using PyPDFLoader
    loader = PyPDFLoader(temp_file_path)
    docs = loader.load()
    os.remove(temp_file_path)

    return docs


# function to creatre vector store
@st.cache_resource 
def create_vector_store(docs_text):
    # Use RecursiveCharacterTextSplitter to split text into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    document_chunks = text_splitter.create_documents([docs_text])

    # create embeddings and vector store
    embeddings = OpenAIEmbeddings()

    try:
        db = FAISS.from_documents(document_chunks, embeddings)
        st.write("Vector store created successfully.")
    except Exception as e:
        st.write(f"Error while creating vector store: {e}")

    return db


# Convert the recorded audio to an AudioData object
def bytes_to_audio_data(audio_bytes):
    audio_file = BytesIO(audio_bytes)
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio_data = recognizer.record(source)
    return audio_data


# Convert voice input to text
def audio_to_text(audio_bytes):
    recognizer = sr.Recognizer()
    try:
        # Convert bytes to AudioData
        audio_data = bytes_to_audio_data(audio_bytes)
        # Use the AudioData to recognize text
        text = recognizer.recognize_google(audio_data, language="ur-PK")
        return text
    except sr.UnknownValueError:
        return "Could not understand audio"
    except sr.RequestError as e:
        return f"Error with the request; {e}"
    

# text to speech conversion using openai tts_1 model
def text_to_speech_conversion(text):
    """Converts text to audio format message using OpenAI's text-to-speech model - tts-1."""
    if text:  # Check if text is not empty
        client = OpenAI()
        response = client.audio.speech.create(
            model="tts-1",  # Model to use for text-to-speech conversion
            voice="fable",  # Voice to use for speech synthesis
            input=text  # Text to convert to speech
        ) 
        
        audio_data = BytesIO(response.read())  # Read the response into a BytesIO object
        
        return audio_data
    

# function to play audio
def play_audio_auto(audio_data, format="audio/webm"):
    """Embeds the audio in HTML with autoplay enabled."""
    # Encode audio data in base64
    audio_base64 = base64.b64encode(audio_data.getvalue()).decode("utf-8")
    audio_html = f"""
        <audio autoplay>
            <source src="data:{format};base64,{audio_base64}" type="{format}">
            Your browser does not support the audio element.
        </audio>
    """
    st.markdown(audio_html, unsafe_allow_html=True)


# streamlit interface
def main():
    st.title('Intelligent Urdu Voice Chatbot')
    with st.sidebar:
        uploaded_file = st.file_uploader('Pl upload a pdf file',accept_multiple_files=False)
    
    
    if uploaded_file is not None:
    # Display the file name
        st.write(f"Uploaded File Name: {uploaded_file.name}")

        docs = process_uploaded_file(uploaded_file)
       
        # combine the text from all pages loaded by document loader
    
        docs_text = "".join([doc.page_content for doc in docs])

        db = create_vector_store(docs_text)

        system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise. Pl answer from the retrieved context only"
        "\n\n"
        "{context}"
        )

        prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
        )


        # create a chain
        document_chain = create_stuff_documents_chain(model, prompt)

        # create a retriever chain by using built in chain
        retriever = db.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever,document_chain)

        # display thr microphone to capture audio from user
        # record for 5 seconds
        audio_bytes = audio_recorder(
        text="Click to record for 5 sec",
        recording_color="red",
        neutral_color="#6aa36f",
        icon_name="microphone-lines",
        icon_size="6x",
        energy_threshold=(-1.0, 1.0),
        pause_threshold=5.0
        )

        if audio_bytes:
            st.audio(audio_bytes, format="audio/wav")
            st.info("Processing audio...")
            input_text = audio_to_text(audio_bytes)
            st.write("Input Text :", input_text)


            response = retrieval_chain.invoke({"input": input_text})
            st.write(response["answer"])
            audio_data = text_to_speech_conversion(response["answer"])

            # If you want to play the audio directly in your Streamlit app:
            st.audio(audio_data, format="audio/webm")
            # Play the audio automatically using HTML
            play_audio_auto(audio_data, format="audio/webm")
        
        

if __name__ == "__main__":
    main()



