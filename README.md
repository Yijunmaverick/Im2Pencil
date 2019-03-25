# Im2Pencil
Pytorch implementation of our CVPR19 [paper](https://arxiv.org/pdf/1903.08682.pdf) on controllable pencil illustration. For non-commercial research collaboration and demostration purposes only.


## Getting started

- Linux
- NVIDIA GPU
- Pytorch 0.4.1
- MATLAB
- [Structured Edge Detection Toolbox](https://github.com/pdollar/edges) by Piotr Dollar 

```
git clone https://github.com/Yijunmaverick/Im2Pencil
cd Im2Pencil
```

## Preparation

- Download the pretrained models:

```
sh pretrained_models/download_models.sh
```

## Testing

  - Test two steps (prediction + generation) together:

```
python test.py  --edge_style 1  --shading_style 1
```


## Citation

```
@inproceedings{Im2Pencil-CVPR-2019,
    author = {Li, Yijun and Fang, Chen and Hertzmann, Aaron and Shechtman, Eli and Yang, Ming-Hsuan},
    title = {m2Pencil: Controllable Pencil Illustration from Photographs},
    booktitle = {IEEE Conference on Computer Vision and Pattern Recognition},
    year = {2019}
}
```

## Acknowledgement

- We express gratitudes to the great work [Pix2Pix](https://phillipi.github.io/pix2pix/) as we benefit a lot from both their paper and code.
