import streamlit as st
import json
import os
import time
import sys
from dotenv import load_dotenv
import requests
import yt_dlp   
from pathlib import Path   

from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.vectorstores import FAISS

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA, LLMChain

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings

import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s') 
logger = logging.getLogger(__name__)    

# Load environment variables
load_dotenv()
api_token = os.getenv("ASSEMBLY_AI_KEY")
if not api_token:
    logger.error("ASSEMBLY_AI_KEY not found in environment variables.")
    sys.exit(1)
os.environ['GOOGLE_API_KEY'] = os.getenv("GOOGLE_API_KEY")  
if not os.environ['GOOGLE_API_KEY']:
    logger.error("GOOGLE_API_KEY not found in environment variables.")
    sys.exit(1)

base_url= "https://api.assemblyai.com/v2"
# Initialize AssemblyAI headers (to accept key and value pairs in JSON format)
headers =   {
    "authorization": api_token,
    "content-type": "application/json"
}

# yt-dlp function to download audio from YouTube video 
def save_audio(url):  
    try:
        # Create temp directory if it doesnt exists
        os.makedirs('temp', exist_ok=True)
        logger.info("Downloading audio from YouTube...") 
        
        ydl_opts = {
            'format': 'bestaudio/best',    #best available audio quality
            'postprocessors': [{           # convert into audio format using FFmpeg library
                'key': 'FFmpegExtractAudio', # extract into audio format using this library
                'preferredcodec': 'mp3',     # desired output format in mp3
                'preferredquality': '192',   #bit rate 192 kbps
            }],
            'outtmpl': 'temp/%(title)s.%(ext)s', # output template and folder for the downloaded file
            'ffmpeg_location': r'C:\Users\Admin\ffmpeg-master-latest-win64-gpl-shared\ffmpeg-master-latest-win64-gpl-shared\bin\ffmpeg.exe',  # specify the location of ffmpeg executable
        }
    
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            audio_filename = ydl.prepare_filename(info).replace('.webm', '.mp3')
            logger.info(f"Audio file saved as: {audio_filename}")
       
        logger.info(f"Audio downloaded successfully.{audio_filename}")
        return Path(audio_filename).name  # Return the path to the downloaded audio file
    except Exception as e:
        logger.error(f"Error downloading audio: {str(e)}")  # storing error in log file
        st.error(f"Failed to download audio from YouTube:{str(e)}")  #show error in Streamlit app
        return None # no file downloaded 
    
# Modify the assemblyai_stt function to return both text and word-level timestamps
# To convert audio into transcript format using AssemblyAI
def assemblyai_stt(audio_filename):
    try:
        audio_path=os.path.join('temp', audio_filename)  # Ensure the path is correct
        if not os.path.exists(audio_path):
            logger.error(f"Audio file {audio_path} does not exist.")
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Upload the audio file to AssemblyAI
        with open(audio_path, 'rb') as f:
            response = requests.post(base_url + "/upload", # sending post requests to assemblyai to upload the audio file
                                      headers=headers, #headers contains key-value pairs for assemblyai to accept authorization
                                        data=f)  #passing the audio file in binary format to get uploaded 
        response.raise_for_status()  # Raise an exception for bad status codes
        
        if response.status_code != 200:
            logger.error(f"Failed to upload audio file: {response.text}")
            st.error("Failed to upload audio file.")
            return None
        
        upload_url = response.json()['upload_url'] #retrieve output in JSON format
        logger.info(f"Audio file uploaded successfully: {audio_path}")

        

        # Start transcription
        data = {
            "audio_url": upload_url
        }
        url = base_url + "/transcript"
        
        response = requests.post(url, headers=headers, json=data) #sending post request to assemblyai to start transcription
        response.raise_for_status()  # Raise an exception for bad status codes
        
        if response.status_code != 200:
            logger.error(f"Failed to start transcription: {response.text}")
            st.error("Failed to start transcription.")
            return None
        logger.info("Transcription started successfully.")
        
        # Get the transcript ID from the response
        
        transcript_id = response.json()['id']
        polling_endpoint = f"{base_url}/transcript/{transcript_id}"

        while True:
            transription_result = requests.get(polling_endpoint, headers=headers).json()  #polling the endpoint to get the transcription result
            
            if transription_result['status'] == 'completed':   
                logger.info("Transcription completed successfully.")
                break
            elif transription_result['status'] == 'error':  
                raise RuntimeError(f"Transcription failed: {transription_result['error']}")
            else:
                time.sleep(3)  # Wait for 3 seconds before polling again

        logger.info("Transcription result received.")
        # retrieve transcription text and word-level timestamps   
        transcription_text = transription_result['text']
        word_timestamps = transription_result['words'] 

        os.makedirs('docs', exist_ok=True)  # Create 'docs' directory if it doesn't exist
        # Save the transcription text to a file 
        with open('docs/transcription.txt', 'w') as file:
            file.write(transcription_text)
        logger.info("Transcription text saved to docs/transcription.txt")

        # Save word-level timestamps to a file
        with open('docs/word_timestamps.json', 'w') as file:
            json.dump(word_timestamps, file)

        logger.info("Successfully transcribed audio eith Word-level timestamps and saved to docs/word_timestamps.json")
        return transcription_text, word_timestamps  # Return both text and word-level timestamps
    except Exception as e:  
        logger.error(f"Error in assemblyai_stt: {str(e)}")
        st.error(f"An error occurred during transcription: {str(e)}")  # Show error in Streamlit app
        return None, None  # Return None if an error occurs : twice None for text and word timestamps
    

# Build RAG System

# Modify the setup_qa_chain function to accept the text and word timestamps
@st.cache_resource  # Cache the QA chain to avoid re-creating it on every run
def setup_qa_chain():
    try:
        # Load the text into a document loader
        loader = TextLoader('docs/transcription.txt')
        documents = loader.load()  # Load the text file into documents

        # Split the text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(documents)

        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")  # Initialize embeddings
        logger.info("Text split into chunks successfully.") 

        # Create a vector store from the chunks
        # FAISS: Facebook AI Simikarity Search is a library for efficient similarity search and clustering of dense vectors
        # It is used to create a vector store from the text chunks for efficient retrieval
        vector_store = FAISS.from_documents(texts, embeddings)  # Create a FAISS vector store from the text chunks

        retriever = vector_store.as_retriever()  # Create a retriever from the vector store
        logger.info("Vector store created successfully.")

        chatmodel=ChatGoogleGenerativeAI(model='gemini-1.5-flash',temperature=0)
        logger.info("Chat model initialized successfully.")
        
        # Create the RetrievalQA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=chatmodel,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
           
        )

        logger.info("QA chain setup successfully.")
        
        with open('docs/word_timestamps.json', 'r') as file:
            word_timestamps = json.load(file)
        logger.info("Word-level timestamps loaded successfully.")

        return qa_chain, word_timestamps  # Return the QA chain and word timestamps
    except Exception as e:
        logger.error(f"Error setting up QA chain: {str(e)}")
        st.error(f"An error occurred while setting up the QA chain: {str(e)}")
        return None, None  # Return None if an error occurs
    
# Function to find relevant word timestamps for a given answer
def find_relevant_timestamps(answer, word_timestamps):
    try:
        relevant_timestamps = []
        answer_words = answer.lower().split()  # Split the answer into words
        for word_info in word_timestamps:
            if word_info['text'].lower() in answer_words:
                relevant_timestamps.append(word_info['start'])
        
        logger.info("Relevant timestamps found successfully.")
        return relevant_timestamps
    except Exception as e:
        logger.error(f"Error finding relevant timestamps: {str(e)}")
        st.error(f"An error occurred while finding relevant timestamps: {str(e)}")
        return []  # Return an empty list if an error occurs    
    
    
# Function to generate summary
def generate_summary(transcription):
    try:
        chatmodel = ChatGoogleGenerativeAI(model='gemini-1.5-flash', temperature=0.7)
        summary_prompt = PromptTemplate(
            input_variables=["transcription"],
            template="Summarize the following transcription in 3-5 sentences:\n\n{transcription}"
        )
        summary_chain = LLMChain(llm=chatmodel, prompt=summary_prompt)
        summary = summary_chain.run(transcription)  # Generate summary using the transcription text
        
        logger.info("Summary generated successfully.")
        return summary   
    except Exception as e:
        logger.error(f"Error generating summary: {str(e)}")
        st.error(f"An error occurred while generating the summary: {str(e)}")
        return None  # Return None if an error occurs
   
# Streamlit app
st.set_page_config(page_title="Audio Enhanced Conversational AI Chatbot", page_icon=":robot:", layout="wide")
st.title("Audio Enhanced Conversational AI Chatbot")        

input_source = st.text_input("Enter YouTube Video URL or Text File Path:", placeholder="https://www.youtube.com/watch?v=example")

# Divide the screen into two columns
if input_source:
    col1, col2 = st.columns(2)
    # Display the uploaded video in the first column
    with col1:
        st.info("Your Uploaded Video ")
        st.video(input_source)  # Display the YouTube video if a valid URL is provided
        audio_filename = save_audio(input_source) 
        if audio_filename:
            # Transcribe the audio file
            transcription, word_timestamps = assemblyai_stt(audio_filename)
            if transcription:
                st.info("Transcription completed successfully. You can now ask questions about the transcription.")
                st.text_area("Transcription Text", transcription, height=300)
                
                # Set up the QA chain
                qa_chain, word_timestamps = setup_qa_chain()

                # Add summary generation option
                if st.button("Generate Summary"):
                    with st.spinner("Generating summary..."):
                        summary = generate_summary(transcription)
                        if summary:
                            st.info("Summary generated successfully.")
                            st.subheader("Summary of Transcription")
                            st.write(summary)
                        else:
                            st.error("Failed to generate summary.")

    # Display the transcription and QA functionality in the second column
    with col2:
        st.info("Chat Below")
        query = st.text_input("Ask a question about the transcription:")
        if query:        
            if qa_chain:    
                with st.spinner("Generating answer..."):
                    try:
                        result = qa_chain({"query": query})  # Run the QA chain with the user's query
                        answer = result['result']  # Extract the answer from the result
                        st.success(answer)  # Display the answer in a success message
                        logger.info(f"Answer generated successfully: {answer}")
                       

                        # Find and display relevant timestamps for the answer
                        relevant_timestamps = find_relevant_timestamps(answer, word_timestamps)
                        if relevant_timestamps:
                            st.subheader("Relevant Word Timestamps")
                            for timestamp in relevant_timestamps[:5]:  # Limit to top 5 timestamps
                                st.write(f"{timestamp // 60}:{timestamp % 60:02d}")  # Display each word with its start and end time
                        else:
                            st.error("QA system is not ready. Please make sure the transcription is completed.")
                        
                    except Exception as e:
                        logger.error(f"Error generating answer: {str(e)}")
                        st.error(f"An error occurred while generating the answer: {str(e)}")

# Clean up temp directory after processing
def cleanup_temp_files():
    """Remove temporary files from the temp directory."""
    if os.path.exists('temp'):
        for file in os.listdir('temp'):
            file_path = os.path.join('temp', file)
            try:
                os.remove(file_path)  # Remove each file in the temp directory
                logger.info(f"Removed temporary file: {file_path}")
            except Exception as e:
                logger.error(f"Error removing temporary file {file_path}: {str(e)}")
