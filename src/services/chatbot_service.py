import os
import glob
import openai
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from dotenv import load_dotenv
import logging

# Configuração de logging para facilitar a depuração
logging.basicConfig(level=logging.INFO)

# Carrega as variáveis de ambiente
load_dotenv()

# Define a chave da API do OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")
if openai.api_key is None:
    raise ValueError("A chave da API do OpenAI não foi fornecida no arquivo .env.")

# Função para extrair texto de todas as páginas do PDF usando PdfReader
def extract_text_from_pdf(pdf_path: str) -> str:
    """Extração de texto de todas as páginas de um PDF usando PdfReader (PyPDF2)"""
    text = ""
    try:
        reader = PdfReader(pdf_path)
        total_pages = len(reader.pages)
        logging.info(f"Iniciando a extração de texto do PDF {pdf_path}. Total de páginas: {total_pages}")
        
        # Itera sobre todas as páginas
        for page_num in range(total_pages):
            page = reader.pages[page_num]
            
            # Extração de texto da página
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
            else:
                logging.warning(f"Não foi possível extrair texto da página {page_num + 1}.")
        
            logging.info(f"Página {page_num + 1} processada com sucesso.")
            
    except Exception as e:
        logging.error(f"Erro ao extrair texto do PDF {pdf_path}: {e}")
    return text

# Função para salvar o texto extraído em um arquivo .txt
def save_text_to_file(text: str, output_path: str):
    """Salva o texto extraído em um arquivo de texto"""
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)
        logging.info(f"Texto extraído salvo em {output_path}")
    except Exception as e:
        logging.error(f"Erro ao salvar texto em arquivo: {e}")

# Função para dividir o texto em blocos menores (chunks)
def create_chunks(text: str) -> list:
    """Divide o texto extraído em pedaços menores para busca eficiente"""
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Função para criar a base de dados vetorial (FAISS)
def create_vectors(pdf_paths: list) -> FAISS:
    """Cria um banco de dados vetorial de embeddings de texto extraído de PDFs"""
    docs = []
    for pdf_path in pdf_paths:
        try:
            text = extract_text_from_pdf(pdf_path)
            # Salva o texto extraído em um arquivo .txt para visualização
            output_txt_path = pdf_path.replace(".pdf", ".txt")
            save_text_to_file(text, output_txt_path)

            if text:
                # Divida o texto extraído em chunks menores
                chunks = create_chunks(text)
                for chunk in chunks:
                    docs.append(Document(page_content=chunk))
        except Exception as e:
            logging.error(f"Erro ao processar o PDF {pdf_path}: {e}")

    # Criação dos embeddings usando OpenAI
    embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key)
    knowledge_base = FAISS.from_documents(docs, embeddings)
    return knowledge_base

# Função para obter a resposta com base na consulta
def get_response_from_query(db, query, k=4):
    """Obtém a resposta para uma consulta, buscando nos documentos da base vetorial"""
    docs = db.similarity_search(query, k=k)
    docs_page_content = "\n".join([d.page_content for d in docs])

    # Define o template do prompt
    chat_template = ChatPromptTemplate.from_messages(
        [
            ("system", """
                Você é um assistente acadêmico especializado em fornecer informações sobre calendários de provas, datas, cronogramas de disciplinas e outros dados acadêmicos para alunos da faculdade ICEV.
                Sua tarefa é buscar essas informações nos documentos fornecidos e responder perguntas relacionadas.
                Caso a consulta seja sobre o calendário de provas ou qualquer outra informação acadêmica, busque essas informações nos documentos e forneça uma resposta precisa, não importando o período.
                Se a informação não estiver disponível nos documentos fornecidos, avise educadamente ao aluno que a informação não foi encontrada.
            """),
            ("user", """
                Pergunta: {pergunta}
                Informações extraídas dos documentos: {docs}
                Resposta: Responda de forma objetiva e clara com base nas informações extraídas dos documentos. Busque por **datas de provas**, **calendários** ou outras informações acadêmicas relevantes.
            """)
        ]
    )

    # Usa o modelo OpenAI Chat (agora com o endpoint correto de chat)
    llm = ChatOpenAI(model="gpt-4", temperature=0.7, openai_api_key=openai.api_key)
    # Se você não tiver acesso ao GPT-4, use o GPT-3.5-turbo:
    # llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, openai_api_key=openai.api_key)
    
    # Criação da cadeia de QA
    chain = load_qa_chain(llm, chain_type="stuff")

    # Executa a cadeia de QA e obtém a resposta, com monitoramento de callback
    with get_openai_callback() as cb:
        response = chain.run(input_documents=docs, question=query)
        print(cb)

    return response

# Função para carregar os caminhos dos PDFs da pasta src/assets/pdfs
def load_pdf_paths(folder_path: str) -> list:
    """Carrega todos os arquivos PDF da pasta especificada"""
    abs_folder_path = os.path.abspath(folder_path)
    pdf_paths = glob.glob(os.path.join(abs_folder_path, "*.pdf"))
    return pdf_paths

# Função principal para processar a consulta
def process_query(query):
    """Processa a consulta de usuário e retorna a resposta do chatbot"""
    # Carrega os caminhos dos PDFs da pasta src/assets/pdfs
    pdf_paths = load_pdf_paths("src/assets/pdfs")
    
    # Cria o banco de dados vetorial
    db = create_vectors(pdf_paths)
    
    # Obtém a resposta do chatbot
    answer = get_response_from_query(db, query)
    
    return {"pergunta": query, "answer": answer}
