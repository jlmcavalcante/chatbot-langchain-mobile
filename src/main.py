from fastapi import FastAPI
from src.controllers import chatbot_controller

app = FastAPI()

app.include_router(chatbot_controller.router)
