from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document  # Importação adicionada

from dotenv import load_dotenv
import os
import glob
import fitz  # PyMuPDF
from youtube_transcript_api import YouTubeTranscriptApi

# Carrega as variáveis de ambiente
load_dotenv()

# Define a chave da API do OpenAI
openai_api_key = os.getenv("OPENAI_API_KEY")

# Cria embeddings usando a API do OpenAI
embeddings = OpenAIEmbeddings(
    openai_api_key=openai_api_key
)

def get_youtube_transcript(video_url: str) -> str:
    # Extrai o ID do vídeo do YouTube
    video_id = video_url.split("v=")[-1]
    
    # Obtém a transcrição do vídeo do YouTube
    transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['pt'])
    transcript_text = " ".join([t['text'] for t in transcript])
    return transcript_text

def create_vectors(video_urls: list, pdf_paths: list) -> FAISS:
    # Cria vetores a partir de vídeos do YouTube e PDFs
    docs = []

    # Processa vídeos do YouTube
    for video_url in video_urls:
        transcript = get_youtube_transcript(video_url)
        if transcript:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            video_docs = text_splitter.split_documents([Document(page_content=transcript)])
            docs.extend(video_docs)

    # Processa PDFs
    for pdf_path in pdf_paths:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        pdf_docs = text_splitter.split_documents([Document(page_content=text)])
        docs.extend(pdf_docs)

    # Cria o banco de dados vetorial
    db = FAISS.from_documents(docs, embeddings)
    return db

def get_response_from_query(db, query, k=4):
    # Obtém resposta da query
    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])

    # Configura o modelo de linguagem
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.7,
        openai_api_key=openai_api_key,
    )

    # Define o template do prompt
    chat_template = ChatPromptTemplate.from_messages(
        [
            (
                "user",
                """
                Você é um assistente especializado em fornecer informações e suporte para pais e responsáveis por crianças autistas.
                Responda a seguinte pergunta de forma clara, compreensiva e prática, considerando o contexto dos cuidadores que buscam entender e ajudar seus filhos autistas.

                Pergunta: {pergunta}
                Informações relevantes: {docs}

                Forneça respostas detalhadas e verbosas, oferecendo sugestões práticas e compreensíveis. Se não souber a resposta, diga "Eu não sei".
                """
            )
        ]
    )

    # Cria a cadeia de LLM
    chain = LLMChain(llm=llm, prompt=chat_template, output_key="answer")

    # Obtém a resposta
    response = chain({"pergunta": query, "docs": docs_page_content})

    return response['answer']

def load_youtube_links(file_path: str) -> list:
    # Carrega links do YouTube a partir de um arquivo de texto
    with open(file_path, 'r') as file:
        links = [line.strip() for line in file.readlines()]
    return links

def load_pdf_paths(folder_path: str) -> list:
    # Carrega caminhos para PDFs em uma pasta
    pdf_paths = glob.glob(os.path.join(folder_path, "*.pdf"))
    return pdf_paths

def process_query(query):
    # Carrega os links do YouTube e os caminhos dos PDFs
    video_urls = load_youtube_links("assets/youtube_links.txt")
    pdf_paths = load_pdf_paths("assets/pdfs")
    
    # Cria vetores para vídeos do YouTube e livros em PDF
    db = create_vectors(video_urls, pdf_paths)
    
    # Obtém a resposta do chatbot
    answer = get_response_from_query(db, query)
    
    return {"pergunta": query, "answer": answer}
