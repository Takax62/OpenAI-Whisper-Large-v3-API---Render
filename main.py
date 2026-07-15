import os
from fastapi import FastAPI, UploadFile, File, Security, HTTPException, status
from fastapi.security.api_key import APIKeyHeader
from faster_whisper import WhisperModel
import io

app = FastAPI()

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
SECRET_API_KEY = os.environ.get("WHISPER_API_KEY", "AlapertelmezettBiztonsagiJelszo123")

# JAVÍTÁS: A hivatalos CTranslate2 modellt töltjük be, ami biztosan elindul CPU-n!
print("🤖 Whisper Base (int8) modell betöltése CPU-ra...")
model = WhisperModel("Systran/whisper-base", device="cpu", compute_type="int8")

async def get_api_key(api_key: str = Security(api_key_header)):
    if api_key == SECRET_API_KEY:
        return api_key
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN, 
        detail="Érvénytelen API Kulcs! Túlterhelés elleni védelem aktív."
    )

@app.get("/")
def home():
    return {"status": "online", "message": "A javított Whisper API sikeresen fut a Renderen!"}

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...), api_key: str = Security(get_api_key)):
    try:
        audio_bytes = await file.read()
        audio_file = io.BytesIO(audio_bytes)
        
        segments, info = model.transcribe(audio_file, language="hu", beam_size=5)
        full_text = "".join([segment.text for segment in segments])
        
        return {"status": "success", "text": full_text.strip()}
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
