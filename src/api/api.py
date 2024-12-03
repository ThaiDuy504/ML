from fastapi import FastAPI, File, UploadFile
import src.model.testpredict as testpredict
import numpy as np
from skimage.io import imread
import os


app = FastAPI()

@app.get("/")
async def read_root():
    return {"message": "Hello World"}

@app.post("/predict/")
async def predict(model, file: UploadFile = File(...)):
    with open("temp.jpg", "wb") as buffer:
        buffer.write(await file.read())
    image = imread("temp.jpg")
    [y_hat,classes] = testpredict.predict(image,model)
    prediction = np.argmax(y_hat)
    os.remove("temp.jpg")
    return {"prediction" : f"{classes[prediction]}","all_probability":f"{y_hat.__str__()}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)





