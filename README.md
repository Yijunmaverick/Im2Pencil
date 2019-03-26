# Im2Pencil
Pytorch implementation of our CVPR19 [paper](https://arxiv.org/pdf/1903.08682.pdf) on controllable pencil illustration. More results and comparisons are shown [here](https://drive.google.com/file/d/1sl5IBD36bMWAvKH7Uz7An0mcrIOmlopv/view). For non-commercial research collaboration and demostration purposes only.

<p>
    <img src='output/t.jpg' width=200 />
    <img src='output/t_outline.png' width=400 />
</p>

A line input (left) and two pencil outline results (middle: clean, right: rough)

<p>
    <img src='output/15--104.png' height=275 />
    <img src='output/15--104_shading.png' height=275 />
</p>

A photo input (left) and four pencil shading results (right: [hatching, crosshatching; blending, stippling])


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

 - Extract the outline and tone image from the input photo (in MATLAB):
 
```
cd extract_edge_tone
Im2Pencil_get_edge_tone.m
```

## Testing

  - Test with different outline and shading styles：

```
python test.py  --outline_style 1  --shading_style 1
```

Outline style: 0 for `rough` and 1 for `clean`

Shading style: 0, 1, 2, 3 for `hatching`, `crosshatching`, `stippling`, and `blending` respectively

For other controllable parameters, check `options/test_options.py`


## Citation

```
@inproceedings{Im2Pencil-CVPR-2019,
    author = {Li, Yijun and Fang, Chen and Hertzmann, Aaron and Shechtman, Eli and Yang, Ming-Hsuan},
    title = {Im2Pencil: Controllable Pencil Illustration from Photographs},
    booktitle = {IEEE Conference on Computer Vision and Pattern Recognition},
    year = {2019}
}
```

## Acknowledgement

- We express gratitudes to the great work [XDoG](http://holgerweb.net/PhD/Research/papers/DoGToonNPAR11.pdf) and [Pix2Pix](https://phillipi.github.io/pix2pix/) as we benefit a lot from both their paper and code.
