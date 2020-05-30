from tqdm import tqdm
from os.path import join, exists
import json
import collections
import tensorflow as tf
import random

random.seed(0)

class ZaloDatasetProcessor(object):
    """ Base class to process & store input data for the Zalo AI Challenge dataset"""
    label_list = ['False', 'True']

    def __init__(self):
        """ ZaloDatasetProcessor constructor
            :parameter val_size: The size of the valelopment set taken from the training set
        """
        self.train_data = []
        self.val_data = []
        self.test_data = []

    def load_from_path(self, dataset_path, mode='train', file_name='train.json', encode='utf-8'):
        """ Load data from file & store into memory
            Need to be called before preprocess(before write_all_to_tfrecords) is called
            :parameter dataset_path: The path to the directory where the dataset is stored
            :parameter encode: The encoding of every dataset file
        """
        mode = mode.lower()
        assert mode in ['train', 'test', 'val', 'squad'], "[Preprocess] Test file mode must be 'zalo' or 'normal'"

        def read_to_inputs(filepath, encode='utf-8', mode='train'):
            """ A helper function that read a json file (Zalo-format) & return a list of input
                :parameter filepath The source file path
                :parameter encode The encoding of the source file
                :parameter mode Return data for training ('normal') or for submission ('zalo')
                :returns A list of input for each data instance, order preserved
            """
            try:
                with open(filepath, 'r', encoding=encode) as file:
                    data = json.load(file)
                    if mode == 'squad':
                        data = data.get('data')
                        res = []
                        for d in data:
                            for par in d.get('paragraphs'):
                                for qas in par.get('qas'):
                                    ques = qas.get('question')
                                    answer = qas.get('answers')
                                    label = qas.get('is_impossible')
                                    if label:
                                        answer = qas.get('plausible_answers')
                                    if answer and len(answer) > 0:
                                        answer = answer[0]
                                        res.append({'question': ques,
                                                     'text': answer.get('text'),
                                                     'label': label})
                        return res

                    else:
                        return [{'question': data_instance['question'],
                                 'text': data_instance['text'],
                                 'label': data_instance.get('label', False)}
                                for data_instance in tqdm(data)]
            except FileNotFoundError:
                return []

        # Get train data, convert to input
        if mode == "train":
            train_data = read_to_inputs(filepath=join(dataset_path, file_name),
                                        encode=encode, mode="train")
            self.train_data.extend(train_data)
        # Get val data, convert to input
        if mode == "val":
            val_data = read_to_inputs(filepath=join(dataset_path, file_name),
                                      encode=encode, mode="val")
            self.val_data.extend(val_data)

        if mode == "test":
            test_data = read_to_inputs(filepath=join(dataset_path, file_name),
                                       encode=encode, mode="test")
            self.test_data.extend(test_data)

        if mode == "squad":
            train_data = read_to_inputs(filepath=join(dataset_path, file_name),
                                        encode=encode, mode="squad")
            self.train_data.extend(train_data)

        # Shuffle training data
        random.shuffle(self.train_data)
        # Shuffle validate data
        random.shuffle(self.val_data)
