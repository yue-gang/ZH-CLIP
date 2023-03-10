# ZH-CLIP: 中文CLIP模型
HuggingFace链接：[nlpcver/zh-clip-vit-roberta-large-patch14](https://huggingface.co/nlpcver/zh-clip-vit-roberta-large-patch14)
## 网络结构
* 图像Encoder网络结构与[openai/clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)相同；
* 文本Encoder网络结构与[hfl/chinese-roberta-wwm-ext-large](https://huggingface.co/hfl/chinese-roberta-wwm-ext-large)相同。
## 效果
### ImageNet-1K zero-shot
| model   | ACC1 |   ACC5 |
| :------------- | :----------: | :------------: |
| nlpcver/zh-clip-vit-roberta-large-patch14 |   53.68%   | 80.05% |
## 使用
### HuggingFace
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
### Other Chinese CLIP Models
此外，在此为了对比不同方法的效果，集成了其他中文CLIP模型的推理方法，为了方便大家使用，也同时把推理代码公开，如有侵权请联系我。代码中仅实现了与clip-vit-large-patch14同级别的模型，后续可能适配更多不同版本模型的使用。
| # | 模型 | 别名 |
| :----: | :---------- | :---------- |
| 0 | [ZH-CLIP](https://github.com/yue-gang/ZH-CLIP) | zhclip |
| 1	| [AltCLIP](https://github.com/FlagAI-Open/FlagAI/tree/master/examples/AltCLIP) | altclip |
| 2	| [Chinese-CLIP](https://github.com/OFA-Sys/Chinese-CLIP)	| cnclip |
| 3	| [TaiyiCLIP](https://github.com/IDEA-CCNL/Fengshenbang-LM)	| taiyiclip |
| 4	| [Multilingual-CLIP](https://github.com/FreddeFrallan/Multilingual-CLIP)	| mclip |
| 5	| [CLIP-Chinese](https://github.com/yangjianxin1/CLIP-Chinese)	| clip-chinese |

使用方法见[inference.py](https://github.com/yue-gang/ZH-CLIP/blob/main/inference.py)
## ToDO
1. 提升模型效果；
2. 更多任务上进行测试；
3. 适配更多模型。
## 致谢
感谢[Yiming Cui](https://ymcui.com/) 在HuggingFace模型处理上的指导。

