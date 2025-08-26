from fastapi import FastAPI

app = FastAPI(title="Smart Research Assistant API")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/")
def root():
    return {"status": "working"}
