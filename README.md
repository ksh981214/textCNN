# textCNN
CNN for Sentence Classification

referenced by [Convolutional Neural Networks for Sentence Classification(Yoon Kim, 2014)](https://arxiv.org/abs/1408.5882)

Korea University Information Retrieval(COSE 472) Assignment7

-----

- CNN-rand : Our baseline model where all words are randomly initialized and then modified during training. 

- CNN-static : A model with pre-trained vectors from word2vec. All words— including the unknown ones that are randomly initialized—are kept static and only the other parameters of the model are learned. 

- CNN-non-static : Same as above but the pre-trained vectors are fine-tuned for each task.

- CNN-multichannel : A model with two sets of word vectors. Each set of vectors is treated as a ‘channel’ and each filter is applied to both channels, but gradients are backpropagated only through one of the channels. Hence the model is able to fine-tune one set of vectors while keeping the other static. Both channels are initialized with word2vec.
