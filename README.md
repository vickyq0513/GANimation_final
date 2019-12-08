
# GANimation: Facial animation from images
### [[Project]](http://www.albertpumarola.com/research/GANimation/index.html)[ [Paper]](https://rdcu.be/bPuaJ) 
Official implementation of [GANimation](http://www.albertpumarola.com/research/GANimation/index.html). In this work we introduce a novel GAN conditioning scheme based on Action Units (AU) annotations, which describe in a continuous manifold the anatomical facial movements defining a human expression. Our approach permits controlling the magnitude of activation of each AU and combine several of them. For more information please refer to the [paper](https://arxiv.org/abs/1807.09251).


![GANimation](http://www.albertpumarola.com/images/2018/GANimation/teaser.png)

## Prerequisites
- Install requirements.txt (Detailed packages and version required are listed)

## Data Preparation
The code requires a directory containing the following files:
- `imgs/`: folder with all image
- `aus_openface.pkl`: dictionary containing the images action units.
- `train_ids.csv`: file containing the images names to be used to train.
- `test_ids.csv`: file containing the images names to be used to test.

An example of this directory is shown in `celeA/`.



## Run
vscode/launch.json: code specifies train and test options

## Reference
```
@article{Pumarola_ijcv2019,
    title={GANimation: One-Shot Anatomically Consistent Facial Animation},
    author={A. Pumarola and A. Agudo and A.M. Martinez and A. Sanfeliu and F. Moreno-Noguer},
    booktitle={International Journal of Computer Vision (IJCV)},
    year={2019}
}
```
