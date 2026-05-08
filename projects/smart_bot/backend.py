from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

# Import the bot from your existing script
from smart_bot_section1 import SmartQABot, setup_environment

app = FastAPI(title="Smart Q&A Bot API")

# Enable CORS so the frontend can talk to the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize bot and environment
setup_environment()
bot = SmartQABot()

@app.get("/api/ask")
async def ask_question(q: str = Query(..., description="The question to ask the bot")):
    """Endpoint to interact with the SmartQABot."""
    response = bot.ask(q)
    return response

@app.get("/")
async def read_index():
    return FileResponse("index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
