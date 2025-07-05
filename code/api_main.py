import fastapi
from transformers import AutoImageProcessor, AutoModel
import torch
import annoy
from torch_model import FeatureExtractionModel
import numpy as np 
import cv2
import pickle
from sklearn.metrics.pairwise import cosine_similarity

dino_prepr = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
dino_model = AutoModel.from_pretrained("facebook/dinov2-small")
annoy_dino = annoy.AnnoyIndex(384, 'angular')
annoy_dino.load("annoy\\test_dino.ann")
with open(".\\embeddings\\dino_embeddings_test.pkl", "rb") as f:
    embeddings_test_dino = pickle.load(f)

custom_model = FeatureExtractionModel()
custom_model.load_state_dict(torch.load("custom_model.pt"))
custom_model.head = torch.nn.Identity()
annoy_custom = annoy.AnnoyIndex(120, "angular")
annoy_custom.load("annoy/test_custom.ann")
with open(".\\embeddings\\custom_embeddings_test.pkl", "rb") as f:
    embeddings_test_custom = pickle.load(f) 

with open("test_paths.pkl", "rb") as f:
    paths = pickle.load(f)

app = fastapi.FastAPI()

@app.post("/recomend_dino/")
async def recomend_dino(file: fastapi.UploadFile = None):
    if not file.content_type.startswith("image/"):
        raise fastapi.HTTPException(status_code=400, detail="File must be an image")
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise fastapi.HTTPException(status_code=400, detail="Could not decode image")

    with torch.no_grad():
        pred = dino_model(**dino_prepr(
            img,
            return_tensors="pt",
        )).pooler_output.numpy()[0]
    nearest = annoy_dino.get_nns_by_vector(pred, 5)
    ret_paths = [paths[i] for i in nearest]
    embeddings = embeddings_test_dino[nearest]
    ret_scores = cosine_similarity(pred[None], embeddings)
    print(ret_scores[0])
    return {"paths": ret_paths, "scores": list(map(float, ret_scores[0]))}