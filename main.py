from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
from transformers import pipeline
import aiofiles
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


app = FastAPI()

# Placeholder for the whisper model's transcription function
def transcribe_audio(file_path: str) -> str:
    # Dummy transcription for the sake of the example
    return "This is a transcribed text from the audio file."

# Summarization pipeline
summarizer = pipeline("summarization", model="t5-base")

class SummaryRequest(BaseModel):
    text: str


@app.post("/transcribe/")
async def transcribe(file: UploadFile = File("sample-O.mp3")):
    try:
    
        # Save the uploaded file
        file_location = f"files/{file.filename}"
        os.makedirs(os.path.dirname(file_location), exist_ok=True)
        
        async with aiofiles.open(file_location, 'wb') as out_file:
            content = await file.read()
            await out_file.write(content)
        
        # Transcribe the audio file
        transcription = transcribe_audio(file_location)
        
        # Summarize the transcription
        summary = summarizer(transcription, max_length=150, min_length=40, do_sample=False)[0]['summary_text']
        
        # Dummy timestamps for the sake of the example
        timestamps = [{"start": 0, "end": 5, "text": "This is a transcribed text from the audio file."}]
        
        # Save transcription, summary, and timestamps locally
        base_filename = os.path.splitext(file.filename)[0]
        with open(f"files/{base_filename}_transcription.txt", 'w') as f:
            f.write(transcription)
        with open(f"files/{base_filename}_summary.txt", 'w') as f:
            f.write(summary)
        with open(f"files/{base_filename}_timestamps.txt", 'w') as f:
            f.write(str(timestamps))
        
        return JSONResponse(content={
            "transcription": transcription,
            "summary": summary,
            "timestamps": timestamps
        }, status_code=200)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.post("/summarize/")
async def summarize_text(request: SummaryRequest):
    try:
        summary = summarizer(request.text, max_length=150, min_length=40, do_sample=False)[0]['summary_text']
        return JSONResponse(content={"summary": summary}, status_code=200)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
