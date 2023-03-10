# Evaluate ImageNet-1k zero-shot classification
## Preprocess data
Download val_images.tar.gz in [ImageNet-1K](https://huggingface.co/datasets/imagenet-1k)
```bash
tar xvzf val_images.tar.gz
cp valprep_new.sh val
sh valprep_new.sh
```
## Evaluate
```bash
python -m eval.imagenet.imagenet_zeroshot
```