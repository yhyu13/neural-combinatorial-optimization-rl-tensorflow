# Neural Combinatorial Optimization with RL

TensorFlow implementation of [Neural Combinatorial Optimization with Reinforcement Learning](http://arxiv.org/abs/1611.09940) for the Traveling Salesman Problem.

![model](./Img/Pointer_Net.png)

The Neural Network consists in a RNN encoder-decoder with an attention module connecting the decoder to the encoder (via a "pointer"). The model is trained by Policy Gradient ([Reinforce](https://link.springer.com/article/10.1007/BF00992696), 1992).

## Requirements

- Python 2.7 or 3.5
- [TensorFlow 1.0.1](https://www.tensorflow.org/install/)
- tqdm

## Architecture

(under progress)

## Usage

- To train a (2D TSP20) model from scratch (data is generated on the fly):
```
> $ python main.py --max_length=20 --inference_mode=False --restore_model=False --save_to=20/repo --log_dir=summary/repo
```

NB: Just make sure ./save/20/repo exists (create it otherwise)

- To visualize training on tensorboard:
```
> $ tensorboard --logdir=summary/repo
```

- To test a trained model:
```
> $ python main.py --max_length=20 --inference_mode=True --restore_model=True --restore_from=20/repo
```

## Results

(under progress)

## Authors

Michel Deudon / [@mdeudon](https://github.com/MichelDeudon)

Pierre Cournut / [@pcournut](https://github.com/pcournut)

## References
Bello, I., Pham, H., Le, Q. V., Norouzi, M., & Bengio, S. (2016). [Neural combinatorial optimization with reinforcement learning](https://arxiv.org/abs/1611.09940). arXiv preprint arXiv:1611.09940.
