import os
import sys
import json
import csv
import glob
import pprint
import numpy as np
import random
import argparse
import pandas as pd
from tqdm import tqdm
from utils import DataProcessor
from utils import SemEvalSingleSentenceExample
from transformers import (
    AutoTokenizer
)


class SemEvalDataProcessor(DataProcessor):
    """Processor for Sem-Eval 2020 Task 4 Dataset.
    Args:
        data_dir: string. Root directory for the dataset.
        args: argparse class, may be optional.
    """

    def __init__(self, data_dir=None, args=None, **kwargs):
        """Initialization."""
        self.args = args
        self.data_dir = data_dir

    def get_labels(self):
        """See base class."""
        return 2  # Binary.

    def _read_data(self, data_dir=None, split="train"):
        """Reads in data files to create the dataset."""
        if data_dir is None:
            data_dir = self.data_dir
        my_dir = os.path.expanduser('~/winter22_cs188_course_project_student/')
        examples = []
        ##################################################
        # TODO: Use csv.DictReader or pd.read_csv to load
        # the csv file and process the data properly.
        # We recommend separately storing the correct and
        # the incorrect statements into two individual
        # `examples` using the provided class
        # `SemEvalSingleSentenceExample` in `utils.py`.
        # e.g. example_1 = ...
        #      example_2 = ...
        #      examples.append(example_1)
        #      examples.append(example_2)
        # For the guid, simply use the row number (0-
        # indexed) for each data instance.
        # raise NotImplementedError("Please finish the TODO!")
        # End of TODO.
        ##################################################
        path = my_dir + 'datasets/semeval_2020_task4/' + split + '.csv'
        print(path)  
        with open(path, newline = '\n') as csvfile:
            spamreader = csv.DictReader(csvfile)
            guid = 0
            for row in spamreader:
                print(row)
                guid = guid
                text_correct = row['Correct Statement']
                text_incorrect = row['Incorrect Statement'] 
                right_reason1 = row['Right Reason1']
                right_reason2 = row ['Right Reason2']
                right_reason3 = row ['Right Reason3']
                confusing_reason1 = row['Confusing Reason1']
                confusing_reason2 = row['Confusing Reason2']
                example_1 = SemEvalSingleSentenceExample(
                        guid=guid,
                        text = text_correct,
                        right_reason1 = right_reason1,
                        right_reason2 = right_reason2,
                        right_reason3 = right_reason3,
                        confusing_reason1 = confusing_reason1,
                        confusing_reason2 = confusing_reason2
                )

                example_2 = SemEvalSingleSentenceExample(
                        guid=guid,
                        text = text_incorrect,
                        right_reason1 = right_reason1,
                        right_reason2 = right_reason2,
                        right_reason3 = right_reason3,
                        confusing_reason1 = confusing_reason1,
                        confusing_reason2 = confusing_reason2      
                )
	   
                examples.append(example_1)
                examples.append(example_2)
                #print(example_1)
                #print(example_2) 
        return examples

    def get_train_examples(self, data_dir=None):
        """See base class."""
        return self._read_data(data_dir=data_dir, split="train")

    def get_dev_examples(self, data_dir=None):
        """See base class."""
        return self._read_data(data_dir=data_dir, split="dev")

    def get_test_examples(self, data_dir=None):
        """See base class."""
        return self._read_data(data_dir=data_dir, split="test")


if __name__ == "__main__":

    # Test loading data.
    proc = SemEvalDataProcessor(data_dir="datasets/semeval_2020_task4")
    train_examples = proc.get_train_examples()
    val_examples = proc.get_dev_examples()
    test_examples = proc.get_test_examples()
    print()
    for i in range(3):
        print(test_examples[i])
    print()
