import numpy as np
import re
import itertools
from collections import Counter


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels(one_data_file, two_data_file, three_data_file, five_data_file, eight_data_file, thirteen_data_file, twentyone_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    one_examples = list(open(one_data_file, "r").readlines())
    one_examples = [s.strip() for s in one_examples]
    two_examples = list(open(two_data_file, "r").readlines())
    two_examples = [s.strip() for s in two_examples]
    three_examples = list(open(three_data_file, "r").readlines())
    three_examples = [s.strip() for s in three_examples]
    five_examples = list(open(five_data_file, "r").readlines())
    five_examples = [s.strip() for s in five_examples]
    eight_examples = list(open(eight_data_file, "r").readlines())
    eight_examples = [s.strip() for s in eight_examples]
    thirteen_examples = list(open(thirteen_data_file, "r").readlines())
    thirteen_examples = [s.strip() for s in thirteen_examples]
    twentyone_examples = list(open(twentyone_data_file, "r").readlines())
    twentyone_examples = [s.strip() for s in twentyone_examples]

    # Split by words
    x_text = one_examples + two_examples + three_examples + five_examples + eight_examples + thirteen_examples + twentyone_examples
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    one_labels = [[1, 0, 0, 0, 0, 0, 0] for _ in one_examples]
    two_labels = [[0, 1, 0, 0, 0, 0, 0] for _ in two_examples]
    three_labels = [[0, 0, 1, 0, 0, 0, 0] for _ in three_examples]
    five_labels = [[0, 0, 0, 1, 0, 0, 0] for _ in five_examples]
    eight_labels = [[0, 0, 0, 0, 1, 0, 0] for _ in eight_examples]
    thirteen_labels = [[0, 0, 0, 0, 0, 1, 0] for _ in thirteen_examples]
    twentyone_labels = [[0, 0, 0, 0, 0, 0, 1] for _ in twentyone_examples]

    y = np.concatenate([one_labels, two_labels, three_labels, five_labels, eight_labels, thirteen_labels, twentyone_labels], 0)
    return [x_text, y]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
