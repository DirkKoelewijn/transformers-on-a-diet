import os
from typing import Union, Callable, Dict, Tuple, List

import xml.etree.ElementTree as Xml
import pandas as pd
import numpy as np

import tensorflow as tf
from sklearn.model_selection import train_test_split


def parse(dataset: str, file: str, validation_split: float = None) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Parses an ABSA XML file in the SemEval-2014, SemEval-2015, SemEval-2016 or MAMS format to a Pandas DataFrame
    containing the id, sentence, target (category), auxiliary sentence and label (polarity). Adds 'none' as label
    for categories not commented on in the sentence, such that each sentence has #categories rows.

    :param dataset: Dataset in {'mams', 'semeval14', 'semeval15', 'semeval16'}
    :param file: Full path to the file
    :param validation_split: Optional. Percentage of sentences to be used in validation. If used, the method will
    return a tuple of 2 DataFrames.
    :return: Pandas DataFrame
    """
    if dataset in ['mams', 'semeval14']:
        data_dict = __parse_se14mams(file)
    elif dataset in ['semeval15', 'semeval16']:
        data_dict = __parse_se1516(file)
    else:
        raise AssertionError(f'Supplied dataset {dataset} is not supported')

    targets = set([str(key) for polarities in data_dict.values() for key in polarities.keys()])

    data = []
    for (sentence_id, sentence), target_dict in data_dict.items():
        for target in targets:
            org_target = str(target)
            target = target.replace('#general', '').replace('anecdotes/miscellaneous', 'anecdotes').replace('_', ' ')
            aspect_target = target.split('#')
            if len(aspect_target) == 2:
                target = f'{aspect_target[1]} of the {aspect_target[0]}'
            data.append([sentence_id, sentence, target, f'what do you think of the {target} of it?',
                         target_dict[org_target] if org_target in target_dict else 'none'])

    df = pd.DataFrame(columns=['id', 'sentence', 'target', 'aux_sentence', 'label'], data=data)

    # Implement validation split
    if validation_split is not None:
        if validation_split > 0.0:
            train, validation = train_test_split(list(df.groupby(by=['id', 'sentence'])), test_size=validation_split)
            return pd.concat([r for k, r in train]), pd.concat([r for k, r in validation])
        else:
            return df, df.iloc[0:0]
    else:
        return df


def __parse_se1516(file: str) -> Dict[Tuple[str, str], Dict[str, str]]:
    """
    Parses an ABSA XML file in the SemEval-2015 or SemEval-2016 format.

    :param file: Full path to the file
    :return: Dictionary mapping (ID, sentence) to {target: polarity}.
    """
    tree = Xml.parse(file)
    data_dict = {}
    for review in tree.findall('Review'):
        for review_sentences in review.findall('sentences'):
            for sentence in review_sentences.findall('sentence'):
                # Each item starts with a sentence ID and the actual sentence
                key = (
                    sentence.get('id').lower().strip(),
                    sentence.find('text').text.lower().strip()
                )
                data_dict[key] = {}

                # Loop over opinions
                for opinions in sentence.findall('Opinions'):
                    for opinion in opinions.findall('Opinion'):
                        data_dict[key][opinion.get('category').lower().strip()] = opinion.get(
                            'polarity').lower().strip()

    return data_dict


def __parse_se14mams(file) -> Dict[Tuple[str, str], Dict[str, str]]:
    """
    Parses an ABSA XML file in the SemEval-2014 or MAMS format.

    :param file: Full path to the file
    :return: Dictionary mapping (ID, sentence) to {target: polarity}.
    """
    tree = Xml.parse(file)
    data_dict = {}
    for i, sentence in enumerate(tree.findall('sentence')):
        key = (
            sentence.get('id').lower().strip() if sentence.get('id') is not None else str(i),
            sentence.find('text').text.lower().strip()
        )
        data_dict[key] = {}

        # Last, add the data for the categories
        for aspect_categories in sentence.findall('aspectCategories'):
            for category in aspect_categories.findall('aspectCategory'):
                data_dict[key][category.get('category').lower().strip()] = category.get(
                    'polarity').lower().strip()
    return data_dict


def balance(
        _df: pd.DataFrame,
        balance_by: str = 'label',
        balance_method: Union[Callable, int, Dict[str, int]] = np.max
) -> pd.DataFrame:
    """
    Balances a dataset.

    :param _df: A Pandas Dataframe containing the parsed and cleaned data with questions.
    :param balance_by: The label of the column to balance the data on.
    :param balance_method: Optional. Function from List[int] to [int] to determine how many of instances of each
    label should be present. balance the polarities. Use `np.max` for oversampling, `np.average` to sample each
    label to the average size and `np.min` for down-sampling.
    :return: Balanced Pandas DataFrame
    """
    # Get spread of balancing variable in dataframe
    spread = {k: rows for k, rows in _df.groupby(by=balance_by)}

    # Check the balance_method input
    if balance_method is None:
        target = {k: len(rows) for k, rows in spread.items()}
    elif callable(balance_method):
        value = round(balance_method([len(value) for key, value in spread.items() if key != 'none']))
        target = {k: value for k in spread}
    elif type(balance_method) == int:
        target = {k: balance_method for k in spread}
    elif type(balance_method) == dict:
        target = {k: balance_method[k] for k in spread}
    else:
        raise AssertionError('balance_method mut be  Callable[list[int] -> int], int or dict[str, int]')

    # Start balancing by sampling within the unique values for the balancing variable
    def assign_repeat_ids(_df: pd.DataFrame, _id: int) -> pd.DataFrame:
        def repeat_id(x: pd.Series, _id=1):
            row = x.copy(deep=True)
            row.id = f'{row.id}_R{_id}'
            return row

        return _df.apply(repeat_id, axis=1, args=(_id,))

    data = []
    for value, rows in spread.items():
        repeats, randoms = target[value] // len(rows), target[value] % len(rows)

        for i in range(repeats):
            data.extend(assign_repeat_ids(rows, i).to_numpy())

        data.extend(assign_repeat_ids(rows.sample(n=randoms), repeats).to_numpy())

    # Get balanced result
    result = pd.DataFrame(columns=_df.columns, data=data).sample(frac=1)

    return result.reset_index(drop=True)


def add_unlabeled(_df: pd.DataFrame, _df_unlabeled: pd.DataFrame, unlabeled_ratio: float = 1.0) -> pd.DataFrame:
    """
    Adds unlabeled data to a DataFrame using the specified unlabeled ratio.
    :param _df: DataFrame with labeled data
    :param _df_unlabeled: DataFrame with all unlabeled data
    :param unlabeled_ratio: Unlabeled data ratio
    :return: Dataframe containing both labeled and unlabeled data
    """
    sampled = _df_unlabeled.sample(n=int(len(_df) * unlabeled_ratio), ignore_index=True)
    sampled['label'] = 'unlabeled'
    return pd.concat([_df, sampled]).sample(frac=1)


def to_dataset(_df: pd.DataFrame, labels: List[str]) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Converts a DataFrame to a dataset that can be used by TensorFlow
    :param _df: DataFrame to convert
    :param labels: Labels occuring in the dataset
    :return: ([normal sentences, auxiliary sentences], one-hot labels) as numpy array
    """
    # First change labels
    label_encoding = {label: i for i, label in enumerate(labels)}
    label_encoding['unlabeled'] = len(labels)
    new_labels = tf.one_hot([label_encoding[label] for label in _df.label], depth=len(labels))
    return [np.array(_df.sentence), np.array(_df.aux_sentence)], new_labels.numpy()


class Preprocessor:
    """
    Class to quickly parse datasets.
    """
    def __init__(self, data_path: str = 'data'):
        """
        :param data_path: Folder in which the dataset folders are present
        """
        self.data_path = data_path

    @staticmethod
    def labels(dataset: str) -> List[str]:
        """
        Returns the label of a given dataset
        :param dataset: Dataset name
        :return: List of labels
        """
        return {
            'mams': ['positive', 'neutral', 'negative', 'none'],
            'semeval14': ['positive', 'neutral', 'negative', 'conflict', 'none']
        }[dataset]

    def parse_train(self,
                    dataset: str,
                    file: str,
                    validation_split: float = None,
                    unlabeled_data: List[Tuple[str, str]] = None,
                    unlabeled_ratio: float = 0.0,
                    create_dataset: bool = True):
        """
        Parses a training dataset
        :param dataset: Dataset name (should correspond to folder)
        :param file: Filename of XML file
        :param validation_split: Part of dataset to output as validation dataset
        :param unlabeled_data: Tuple of dataset > files to use as unlabeled data
        :param unlabeled_ratio: Ratio of unlabeled data to use compared to labeled data
        :param create_dataset: Whether to create a dataset. If false, output will be a DataFrame.
        :return: Numpy arrays or Dataframe
        """
        # Retrieve
        df_train = parse(dataset, os.path.join(self.data_path, dataset, file), validation_split=validation_split)
        df_validation = None
        if type(df_train) == tuple:
            df_validation = df_train[1]
            df_train = df_train[0]

        # Balance
        df_train = balance(df_train)

        # Add unlabeled data
        if unlabeled_ratio > 0 and unlabeled_data is not None:
            dfs = []
            for ds, f in unlabeled_data:
                dfs.append(parse(ds, os.path.join(self.data_path, ds, f)))

            df_train = add_unlabeled(df_train, pd.concat(dfs), unlabeled_ratio=unlabeled_ratio)

        # Return as dataset
        if create_dataset:
            df_train = to_dataset(df_train, self.labels(dataset))
            if df_validation is not None:
                df_validation = to_dataset(df_validation, self.labels(dataset))

        return df_train if df_validation is None else (df_train, df_validation)

    def parse_test(self, dataset: str, file: str, create_dataset: bool = True):
        """
        Parses a testing dataset
        :param dataset: Dataset name (should correspond to folder)
        :param file: Filename of XML file
        :param create_dataset: Whether to create a dataset. If false, output will be a DataFrame.
        :return: Numpy arrays or Dataframe
        """
        res = parse(dataset, os.path.join(self.data_path, dataset, file))
        return to_dataset(res, self.labels(dataset)) if create_dataset else res
