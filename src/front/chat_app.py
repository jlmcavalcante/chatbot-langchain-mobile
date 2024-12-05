import sys
import os
import streamlit as st
from PyPDF2 import PdfReader
from streamlit_chat import message  # Importando a função para mensagens de chat

# Adiciona o diretório raiz ao sys.path para garantir que o Python encontre os módulos dentro de src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Importando a função process_query do back-end
from src.services.chatbot_service import process_query  # Seu código de back-end para consultas

# Função de interação com o modelo (chama a função process_query para obter a resposta)
def get_bot_response(user_input):
    response = process_query(user_input)
    return response["answer"]  # Apenas o valor da chave "answer"

# Função para salvar arquivos PDF na pasta assets/pdfs
def save_uploaded_file(uploaded_file):
    file_path = os.path.join("src", "assets", "pdfs", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

# Função para processar o PDF e extrair o conteúdo
def process_pdf_for_bot(pdf_path):
    pdf_reader = PdfReader(pdf_path)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()  # Extrai o texto de cada página do PDF
    return text

def main():
    st.set_page_config(page_title="Chat Acadêmico", layout="wide")
    st.title("Chat Acadêmico - ICEV")

    # Armazenamento das mensagens
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Adiciona CSS para a área de chat
    st.markdown(
        """
        <style>
            .chat-container {
                height: 400px;  /* Defina a altura desejada */
                overflow-y: auto;  /* Permite a rolagem */
                padding: 10px;
                background-color: #f9f9f9;
                border-radius: 8px;
                margin-bottom: 15px;
            }
            .input-container {
                position: sticky;
                bottom: 0;
                padding: 10px;
                background-color: #fff;
                border-radius: 8px;
                box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
                z-index: 100;
            }
        </style>
        """, unsafe_allow_html=True
    )

    # Criação de duas colunas para o input e o upload de documentos
    col1, col2 = st.columns([5, 2])  # Coluna 1 maior para o input, coluna 2 menor para o upload

    with col1:
        # Container para o histórico de mensagens com rolagem
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        if not st.session_state.messages:
            message("Olá, como posso ajudar você?", key="welcome_message")
        for idx, msg in enumerate(st.session_state.messages):
            if msg["role"] == "user":
                message(msg["content"], is_user=True, key=f"user_{idx}")
            else:
                message(msg["content"], key=f"bot_{idx}")
        st.markdown('</div>', unsafe_allow_html=True)

        # Caixa de entrada fixa na parte inferior
        with st.container():
            user_input = st.text_input("Digite sua pergunta:", "", key="user_input")
            if st.button("Enviar"):
                if user_input:
                    # Adiciona a pergunta do usuário ao histórico de mensagens
                    st.session_state.messages.append({"role": "user", "content": user_input})

                    # Obtém a resposta do bot (chama o back-end)
                    bot_response = get_bot_response(user_input)

                    # Adiciona a resposta do bot ao histórico de mensagens
                    st.session_state.messages.append({"role": "bot", "content": bot_response})

                    # Limpa o campo de entrada após a resposta
                    st.session_state.user_input = ""  # Limpa a entrada do usuário

                    # Recarrega a interface para mostrar o histórico atualizado
                    st.experimental_rerun()

                else:
                    st.warning("Por favor, insira uma pergunta.")
    
    with col2:
        # Adicionando o uploader de arquivo na coluna lateral
        uploaded_file = st.file_uploader("Escolha um arquivo PDF", type=["pdf"])

        if uploaded_file is not None:
            # Salva o arquivo PDF na pasta assets/pdfs
            file_path = save_uploaded_file(uploaded_file)
            st.success(f"Arquivo {uploaded_file.name} carregado com sucesso!")

            # Processa o PDF para ser usado no bot
            pdf_text = process_pdf_for_bot(file_path)

            # Adiciona o conteúdo do PDF ao histórico de mensagens
            st.session_state.messages.append({"role": "bot", "content": f"Conteúdo extraído do PDF: {pdf_text[:500]}..."})

            # Atualiza o histórico com o conteúdo extraído
            st.experimental_rerun()

# Executar a aplicação
if __name__ == "__main__":
    main()
