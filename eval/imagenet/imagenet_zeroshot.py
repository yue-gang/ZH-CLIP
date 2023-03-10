import sys
import torch
import torchvision.datasets as datasets
from . import zeroshot_classification
from .imagenet_templates_zh import imagenet_classnames, openai_imagenet_template
from model_wrapper.get_model import get_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_imagenet(imagenet_val, transform):
    data_path = imagenet_val
    dataset = datasets.ImageFolder(data_path, transform=transform)
    sampler = None
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=100,
        num_workers=1,
        sampler=sampler,
    )
    return dataloader

def evaluate(imagenet_valid_path, model):
    dataloader = get_imagenet(imagenet_valid_path, model.transform)
    zeroshot_templates = openai_imagenet_template
    classnames = imagenet_classnames
    metrics = zeroshot_classification.evaluate(
        model,
        dataloader,
        model.tokenizer,
        classnames,
        zeroshot_templates,
        device=next(model.parameters()).device,
        amp=True,
    )
    result = {
        "dataset": 'imagenet',
        "metrics": metrics
    }
    return result

if __name__ == "__main__":
    imagenet_valid_path = '/data/imagenet/val'
    model_name = 'zhclip'
    model = get_model(model_name)
    assert model is not None
    model = model.eval().to(device)
    result = evaluate(imagenet_valid_path, model)
    print(result)
