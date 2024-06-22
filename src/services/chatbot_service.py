from langchain.document_loaders import YoutubeLoader, PDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate

from dotenv import load_dotenv
import os
import glob

# Carrega as variáveis de ambiente
load_dotenv()

# Define a chave da API do OpenAI
openai_api_key = os.getenv("OPENAI_API_KEY")

# Cria embeddings usando a API do OpenAI
embeddings = OpenAIEmbeddings(
    openai_api_key=openai_api_key
)

def create_vector_from_yt_urls(video_urls: list) -> FAISS:
    # Cria vetores a partir de vídeos do YouTube
    docs = []
    for video_url in video_urls:
        loader = YoutubeLoader.from_youtube_url(video_url, language="pt")
        transcript = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        video_docs = text_splitter.split_documents(transcript)
        docs.extend(video_docs)
    db = FAISS.from_documents(docs, embeddings)
    return db

def create_vector_from_pdfs(pdf_paths: list) -> FAISS:
    # Cria vetores a partir de PDFs
    docs = []
    for pdf_path in pdf_paths:
        loader = PDFLoader(pdf_path)
        pdf_docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs.extend(text_splitter.split_documents(pdf_docs))
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
                Você é um assistente para pais ou responsáveis de pessoas com autismo que responde perguntas baseado em transcrições de vídeos do YouTube e textos de livros em PDF.
                Responda a seguinte pergunta: {pergunta}
                Procurando nos seguintes documentos: {docs}

                Use somente a informação dos documentos para responder a pergunta. Se você não sabe, responda com "Eu não sei".
                Suas respostas devem ser bem detalhadas e verbosas.
                """
            )
        ]
    )

    # Cria a cadeia de LLM
    chain = LLMChain(llm=llm, prompt=chat_template, output_key="answer")

    # Obtém a resposta
    response = chain({"pergunta": query, "docs": docs_page_content})

    return response

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
    video_urls = load_youtube_links("../../assets/youtube_links.txt")
    pdf_paths = load_pdf_paths("../../assets/pdfs")
    
    # Cria vetores para vídeos do YouTube
    db_videos = create_vector_from_yt_urls(video_urls)
    
    # Cria vetores para livros em PDF
    db_pdfs = create_vector_from_pdfs(pdf_paths)
    
    # Combina os bancos de dados vetoriais
    db_combined = FAISS.merge([db_videos, db_pdfs])
    
    # Obtém a resposta do chatbot
    response = get_response_from_query(db_combined, query)
    
    return response
