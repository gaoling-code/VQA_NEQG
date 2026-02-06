# Learning neighbor-enhanced region representations and question-guided visual representations for visual question answering

This repository contains the code for the paper "Learning neighbor-enhanced region representations and question-guided visual representations for visual question answering". 

## Abstract

Great strides have been made in visual question answering field (VQA) based on the application and development of deep learning in related research fields. Existing models in this field focus on the learning and fusion of visual and textual features. However, it is extremely crucial for VQA tasks to focus on the associations between image regions and use question information to enhance key features. In this paper, we propose a method for mining and integrating neighbor-enhanced region representations and question-guided visual representations. Particularly, the region feature graph is first constructed to integrate the features of all regions and the relationships between regions. Secondly, a random walk-based method is presented to acquire the neighbor-enhanced region representations, which combines the topological relationships of all region nodes in the graph. The question-guided vertical and horizontal dual attention mechanism is then proposed to enhance the region representation from the region level and the feature level, respectively. Finally, the enhanced region representation and question representation are integrated adaptively to achieve answer prediction. Convincible experiments show that our method achieves improvements and outperforms prior state-of-the-art methods on two competitive benchmarks, i.e., VQA v1 and VQA v2.

## Keywords

Visual question answering; Deep learning; Feature graph; Attention mechanism; Random walk

## Requirements

Install all required python dependencies:

```
pip install -r requirements.txt
```

## Datasets

The models are trained and evaluated on two open-source datasets:
- VQA v1 dataset:
- VQA v2 dataset:
  Accessible at: [visualqa](https://visualqa.org/download.html).

## Data Pre-processing

python preprocess-images.py
python preprocess-vocab.py

## Usage

To train the model by yourself, please run `python train.py --trainval`.

To test the model, please run `python train.py --test --resume=the path of the model.pth`.


## Acknowledgements 

We highly thank "Making the V in VQA Matter: Elevating the Role of Image Understanding in Visual Question Answering". [paper](https://arxiv.org/abs/1612.00837) and "VQA: Visual Question Answering". [paper](https://arxiv.org/abs/1505.00468)


## Reference
```
@article{Ling_2024_Learning,
  title={Learning neighbor-enhanced region representations and question-guided visual representations for visual question answering},
  author={Ling Gao, Hongda Zhang, Nan Sheng, Lida Shi, Hao Xu},
  journal={Expert Systems with Applications},
  year={2024}
}
```


