from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
from src.model.model import MODEL_LIST
import src.model.testpredict as testpredict
from skimage.io import imread


app = FastAPI()

@app.get("/")
async def read_root():
    return {"message": "Hello World"}




for model in MODEL_LIST:
    @app.post(f"/predict/{model}/")
    async def predict(file: UploadFile = File(...),model=model):
        with open("temp.jpg", "wb") as buffer:
            buffer.write(await file.read())
        image = imread("temp.jpg")
        [y_hat,classes] = testpredict.predict(image,model)
        return {"prediction" : f"{classes[int(y_hat[0])]}"}

# @app.post("/predict/")
# async def predict(file: UploadFile = File(...)):
#     with open("temp.jpg", "wb") as buffer:
#         buffer.write(await file.read())
#     image = imread("temp.jpg")
#     [y_hat,classes] = testpredict.predict(image)
#     return {"prediction" : f"{classes[int(y_hat[0])]}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)





