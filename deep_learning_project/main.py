from fastapi import FastAPI
from routs.data import router as process_data_router
from routs.train import router as train_router
from routs.predict import router as predict_router
import uvicorn

app = FastAPI()

app.include_router(process_data_router)
app.include_router(train_router)
app.include_router(predict_router)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
