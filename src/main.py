from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from src.controllers import chatbot_controller

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite todas as origens, ajuste conforme necessário
    allow_credentials=True,
    allow_methods=["*"],  # Permite todos os métodos HTTP (GET, POST, etc)
    allow_headers=["*"],  # Permite todos os headers
)

app.include_router(chatbot_controller.router)
