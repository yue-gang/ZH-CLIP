# ZH-CLIP: 中文CLIP模型
## 网络结构
* 图像Encoder网络结构与[openai/clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)相同；
* 文本Encoder网络结构与[hfl/chinese-roberta-wwm-ext-large](https://huggingface.co/hfl/chinese-roberta-wwm-ext-large)相同。
## 效果
### ImageNet-1K zero-shot
| model   | ACC1 |   ACC5 |
| :------------- | :----------: | :------------: |
| nlpcver/zh-clip-vit-roberta-large-patch14 |   53.68%   | 80.05% |
## 使用
```python
from PIL import Image
import requests
from models.zhclip import ZhCLIPProcessor, ZhCLIPModel  # From https://www.github.com/nlpcver/Zh-CLIP

version = 'nlpcver/zh-clip-vit-roberta-large-patch14'
model = ZhCLIPModel.from_pretrained(version)
processor = ZhCLIPProcessor.from_pretrained(version)

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
inputs = processor(text=["一只猫", "一只狗"], images=image, return_tensors="pt", padding=True)

outputs = model(**inputs)
image_features = outputs.image_features
text_features = outputs.text_features
text_probs = (image_features @ text_features.T).softmax(dim=-1)
```
## 致谢
感谢[Yiming Cui](https://ymcui.com/) 在HuggingFace模型处理上的指导。

