# Integrated Markov Chain
Official implementation of Integrated Markov Chain for Transformer Interpretability.

We provide a [jupyter notebook](./tutorials.ipynb) for quickly experience the visualization of our approach, as shown in the figure.
![fig1](images/fig1.png)

## Introduction

We introduce an explainability method which is able to visualize classifications made by a transformer-based model. You can enter a specified category to see the evidence the model is classified as this category.

The pipeline of our proposed method **Integrated Markov Chain** is as followed:

![pipeline](./images/pipeline.jpg)


## Credits
ViT implementation is based on:
- <https://github.com/hila-chefer/Transformer-Explainability>
- <https://github.com/rwightman/pytorch-image-models>
- <https://github.com/lucidrains/vit-pytorch>
- Pretrained weights from: <https://github.com/google-research/vision_transformer>

Evaluation experiments is based on:
- [Perturbation test](https://github.com/hila-chefer/Transformer-Explainability)
- [Deletion & Insertion](https://github.com/eclique/RISE)
- [Energy-based pointing game](https://github.com/haofanwang/Score-CAM)
- [Weakly-Supervised Semantic Segmentation](https://github.com/OFRIN/PuzzleCAM)

## Reproducing evaluation results

### Deletion & Insertion
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=./:$PYTHONPATH python3 baselines/ViT/del_ins.py --method integrated_markov_chain
    
You can using the `--method` argument to choose the explainability method you want. The `--num_samples` argument is used to set the number of test samples (default: 2000). The `--batch_size` argument is used to set the batch size (default: 8).

### Energy-based Pointing Game
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=./:$PYTHONPATH python3 baselines/ViT/energy_point_game.py --method integrated_markov_chain
    
You can using the `--method` argument to choose the explainability method you want. The `--num_samples` argument is used to set the number of test samples (default: 2000). The `--batch_size` argument is used to set the batch size (default: 8).

### Perturbation Test
    # step 1:
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=./:$PYTHONPATH python3 baselines/ViT/generate_visualizations.py --method integrated_markov_chain --imagenet-validation-path /path/to/imagenet_validation_directory
    
    # step 2:
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=./:$PYTHONPATH python3 baselines/ViT/pertubation_eval_from_hdf5.py --method integrated_markov_chain

You can using the `--method` argument to choose the explainability method you want.

### Segmentation Results

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=./:$PYTHONPATH python3 baselines/ViT/imagenet_seg_eval.py --method integrated_markov_chain --imagenet-seg-path /path/to/gtsegs_ijcv.mat
    
You can using the `--method` argument to choose the explainability method you want. You must provide a path to imagenet segmentation data in `--imagenet-seg-path`.

% ### Weakly-Supervised Semantic Segmentation


## Citation





