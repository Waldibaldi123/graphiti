from fastapi import FastAPI

app = FastAPI()

@app.get("/process")
def process_string(text: str) -> str:
    print(text)
    return text

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

