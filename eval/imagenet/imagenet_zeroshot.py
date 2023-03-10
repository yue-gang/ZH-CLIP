import sys
import torch
import argparse
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--val-path",
        type=str,
        default= '/data/imagenet/val',
        help="ImageNet-1K Valid File."
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default= 'zhclip',
        help="Chinese Clip Models, Choose From zhclip, altclip, chclip, taiyiclip, mclip, clip_chinese"
    )
    args = parser.parse_args()
    assert args.model_name in {'zhclip', 'altclip', 'chclip', 'taiyiclip', 'mclip', 'clip_chinese'}
    model = get_model(args.model_name)
    model = model.eval().to(device)
    result = evaluate(args.val_path, model)
    print(result)

if __name__ == "__main__":
    main()
