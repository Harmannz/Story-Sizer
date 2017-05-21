# Story-Sizer (WIP)
A convolutional neural network (cnn) that estimates the size of agile stories based on its description.
The data that the cnn uses is not present as it contains sensitive information to my office project.

## Getting Started
This project requires Anaconda installed on your machine.

Use the environment.yml file in environment folder to import the conda environment for this project.
`conda env create -f environment.yml`

Then run train_stories.py and supply arguments as needed.
```bash
./train_stories.py --help
```

You will need to add your own data files so ensure that you update the code as required.


## Reference

This code is based on:

**["Implementing a CNN for Text Classification in Tensorflow" blog post.](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/)**

Is slightly simplified implementation of Kim's [Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1408.5882) paper in Tensorflow.

