# main.py

'''
To run the website -- 'uvicorn main:app --reload' 

'''


from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from langchain_ollama import ChatOllama
import os
import uuid
import model_dev

app = FastAPI()

# --- Setup Directories ---
STATIC_DIR = "static"
UPLOADS_DIR = os.path.join(STATIC_DIR, "uploads")
os.makedirs(UPLOADS_DIR, exist_ok=True) # Create the folder if it doesn't exist

# Correctly mount ALL static directories
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/media", StaticFiles(directory="media"), name="media") # This line fixes the background image issue
templates = Jinja2Templates(directory="templates")

# --- Load ML Model and LLM on Startup ---
try:
    model, class_names = model_dev.load_dependencies()
    llm = ChatOllama(model="gemma3:1b")
except Exception as e:
    print(f"Error loading models: {e}")
    model, class_names, llm = None, None, None

# --- Image Prediction Endpoints ---

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Renders the main upload page."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def upload_and_predict(file: UploadFile = File(...)):
    """Handles file processing and redirects to the results page."""
    if not model or not class_names:
        return RedirectResponse(url="/?error=Image+Model+not+loaded", status_code=303)

    contents = await file.read()
    
    file_extension = file.filename.split('.')[-1]
    unique_filename = f"{uuid.uuid4()}.{file_extension}"
    file_path = os.path.join(UPLOADS_DIR, unique_filename)
    with open(file_path, "wb") as f:
        f.write(contents)
    
    prediction = model_dev.predict_from_bytes(model, class_names, contents)
    
    return RedirectResponse(
        url=f"/results?prediction={prediction}&image_name={unique_filename}",
        status_code=303
    )

@app.get("/results", response_class=HTMLResponse)
async def show_results(request: Request, prediction: str, image_name: str):
    """Displays the prediction results."""
    image_url = request.url_for('static', path=f'uploads/{image_name}')
    
    return templates.TemplateResponse("index.html", {
        "request": request,
        "prediction": prediction,
        "image_url": image_url
    })

# --- Chatbot Endpoints (Restored) ---

from fastapi import Query

@app.get("/chat", response_class=HTMLResponse)
def chatbot(request: Request, user_message: str = Query(None)):
    """Displays chatbot page or auto-asks a question if provided."""
    bot_response_content = None
    if user_message:
        if not llm:
            bot_response_content = "Sorry, the chatbot model is not loaded."
        else:
            response = llm.invoke(user_message)
            bot_response_content = response.content

    return templates.TemplateResponse(
        "chatbot.html",
        {
            "request": request,
            "user_message": user_message,
            "bot_response": bot_response_content
        }
    )

@app.post("/chat", response_class=HTMLResponse)
def chat(request: Request, user_message: str = Form(...)):
    """Handles the chatbot conversation."""
    if not llm:
        bot_response_content = "Sorry, the chatbot model is not loaded."
    else:
        response = llm.invoke(user_message)
        bot_response_content = response.content

    return templates.TemplateResponse(
        "chatbot.html",
        {
            "request": request,
            "user_message": user_message,
            "bot_response": bot_response_content
        }
    )