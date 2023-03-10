import sys
import torch
from PIL import Image
from model_wrapper.get_model import get_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def inference(model, img_path, texts):
    image = Image.open(img_path)
    image_inputs = model.transform(image, return_tensors='pt').to(device)
    text_inputs = model.tokenizer(texts, padding=True, truncation=True,
                                  max_length=77, return_tensors='pt').to(device)
    with torch.no_grad(), torch.cuda.amp.autocast():
        image_features = model.get_image_features(**image_inputs)
        text_features = model.get_text_features(**text_inputs)
        text_probs = (image_features @ text_features.T).softmax(dim=-1)
    return text_probs.cpu().numpy()

if __name__=="__main__":
    model_name = 'zhclip'
    model = get_model(model_name)
    assert model is not None
    model = model.to(device).eval()
    result = inference(model, './images/dog.jpeg', ['一只狗', '一只猫', '一只狼', '狗狗'])
    print(result)