from tqdm import tqdm
from os.path import join, exists
import json
import collections
import tensorflow as tf
import random

random.seed(0)


class InputExample(object):
    """A single training/test example in Zalo format for simple sequence classification."""

    def __init__(self, guid, question, text, title=None, label=None):
        """ Constructs a InputExample.
            :parameter guid: Unique id for the example.
            :parameter question: The untokenized text of the first sequence.
            :parameter text (Optional): The untokenized text of the second sequence
            :parameter label (Optional): The label of the example. This should be
            :parameter title (Optinal): The Wikipedia title where the text is retrieved
        """
        self.guid = guid
        self.question = question
        self.text = text
        self.title = title
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 guid,
                 input_ids,
                 input_mask,
                 segment_ids,
                 label_id,
                 is_real_example=True):
        self.guid = guid
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.is_real_example = is_real_example


class PaddingInputExample(object):
    """ Fake example so the num input examples is a multiple of the batch size.

        When running eval/predict on the TPU, we need to pad the number of examples
        to be a multiple of the batch size, because the TPU requires a fixed batch
        size. The alternative is to drop the last batch, which is bad because it means
        the entire output data won't be generated.

        We use this class instead of `None` because treating `None` as padding
        battches could cause silent errors.
    """


class ZaloDatasetProcessor(object):
    """ Base class to process & store input data for the Zalo AI Challenge dataset"""
    label_list = ['False', 'True']

    def __init__(self):
        """ ZaloDatasetProcessor constructor
            :parameter dev_size: The size of the development set taken from the training set
        """
        self.train_data = []
        self.dev_data = []
        self.test_data = []

    def load_from_path(self, dataset_path, mode='train', encode='utf-8'):
        """ Load data from file & store into memory
            Need to be called before preprocess(before write_all_to_tfrecords) is called
            :parameter dataset_path: The path to the directory where the dataset is stored
            :parameter train_filename: The name of the training file
            :parameter test_filename: The name of the test file
            :parameter dev_filename: The name of the development file
            :parameter train_augmented_filename: The name of the augmented training file
            :parameter testfile_mode: The format of the test dataset (either 'zalo' or 'normal' (same as train set))
            :parameter encode: The encoding of every dataset file
        """
        mode = mode.lower()
        assert mode in ['train', 'test', 'dev'], "[Preprocess] Test file mode must be 'zalo' or 'normal'"

        def read_to_inputs(filepath, encode='utf-8', mode='train'):
            """ A helper function that read a json file (Zalo-format) & return a list of InputExample
                :parameter filepath The source file path
                :parameter encode The encoding of the source file
                :parameter mode Return data for training ('normal') or for submission ('zalo')
                :returns A list of InputExample for each data instance, order preserved
            """
            try:
                with open(filepath, 'r', encoding=encode) as file:
                    data = json.load(file)
                if mode in ['test', 'dev']:
                    returned = []
                    for data_instance in tqdm(data):
                        returned.extend({'question': data_instance['question'],
                                         'text': paragraph_instance['text'],
                                         'label': paragraph_instance.get('label', None)}
                                        for paragraph_instance in data_instance['paragraphs'])
                    return returned
                else:
                    return [{'question': data_instance['question'],
                             'text': data_instance['text'],
                             'label': data_instance.get('label', False)}
                            for data_instance in tqdm(data)]
            except FileNotFoundError:
                return []

        # Get train data, convert to InputExamples
        train_data = []
        if mode == "train":
            train_data = read_to_inputs(filepath=join(dataset_path, "train.json"),
                                        encode=encode)
            self.train_data.extend(train_data)
        # Get dev data, convert to InputExample
        if mode == "dev":
            dev_data = read_to_inputs(filepath=join(dataset_path, "dev.json"),
                                      encode=encode)
            self.dev_data.extend(dev_data)

        if mode == "test":
            test_data = read_to_inputs(filepath=join(dataset_path, "test.json"),
                                       encode=encode)

        # Shuffle training data
        random.shuffle(self.train_data)
