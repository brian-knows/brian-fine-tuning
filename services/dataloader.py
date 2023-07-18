import json
import random
from datasets import Dataset
from sklearn.model_selection import train_test_split

class DataLoader:

    def __init__(self):
        pass

    def load_dataset_from_local(self, path, split=False, test_size=0.3):
        if not split:
            with open(path, 'r') as dataset:
                dataset = json.load(path)
            return dataset

        if split:
            with open(path, 'r') as dataset:
                dataset_lines = dataset.readlines()
                train, test = train_test_split(dataset_lines, test_size=test_size)
            return train, test


    def create_hf_dataset(self, input_json: dict):
        """
        Takes in input a dictionary
        :return: Hugging Face Dataset object
        """

        dataset = Dataset.from_dict(input_json)
        return dataset




