## Convolutional Neural Networks for Sentence Classification
Code for reproducing all results for all datasets by the paper [Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1408.5882) (EMNLP 2014).

### Requirements
Code is written in Python (3.5) and requires Theano (1.0.2). We provide all datasets, and the implementation supports multi-class, as well as cross validation and pre-split datasets.

Using the pre-trained `word2vec` vectors will also require downloading the binary file from
https://code.google.com/p/word2vec/


### Data Preprocessing
To process the raw data, run

```
python process_data.py
```

where path points to the word2vec binary file (i.e. `GoogleNews-vectors-negative300.bin` file).
This will create a pickle object called `mr.p` in the same folder, which contains the dataset
in the right format.

### Running the models (CPU)
Example commands:

```
THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 python3.5 conv_net_sentence.py -nonstatic -rand
THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 python3.5 conv_net_sentence.py -static -word2vec
THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 python3.5 conv_net_sentence.py -nonstatic -word2vec
```

This will run the CNN-rand, CNN-static, and CNN-nonstatic models respectively in the paper.

### Using the GPU
GPU will result in a good 10x to 20x speed-up, so it is highly recommended.
To use the GPU, simply change `device=cpu` to `device=gpu` (or whichever gpu you are using).
For example:
```
THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32 python3.5 conv_net_sentence.py -nonstatic -word2vec
```

### Other Implementations
#### TensorFlow
[Denny Britz](http://www.wildml.com) has an implementation of the model in TensorFlow:

https://github.com/dennybritz/cnn-text-classification-tf

He also wrote a [nice tutorial](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow) on it, as well as a general tutorial on [CNNs for NLP](http://www.wildml.com/2015/11/understanding-convolutional-neural-networks-for-nlp).

#### Torch
[HarvardNLP](http://harvardnlp.github.io/) group has an implementation in Torch.

https://github.com/harvardnlp/sent-conv-torch

### Hyperparameters
At the time of my original experiments I did not have access to a GPU so I could not run a lot of different experiments.
Hence the paper is missing a lot of things like ablation studies and variance in performance, and some of the conclusions
were premature (e.g. regularization does not always seem to help).

Ye Zhang has written a [very nice paper](http://arxiv.org/abs/1510.03820) doing an extensive analysis of model variants (e.g. filter widths, k-max pooling, word2vec vs Glove, etc.) and their effect on performance.
