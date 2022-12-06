import os
from typing import List, Iterator

import tensorflow as tf
import tensorflow_addons as tfa
# noinspection PyUnresolvedReferences
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint

from src.data import Preprocessor
from src.models import BaselineModel, BaseGAN, WGAN
from src.callbacks import EvaluateCallback, GANCallback, WGANCallback


def get_config():
    return {
        'mams': {
            'train': 'train.xml',
            'val': 'validation.xml',
            'test': 'test.xml',
            'unsupervised': [
                ('semeval14', 'train.xml'),
                ('semeval15', 'train.xml'),
                ('semeval16', 'train.xml')
            ],
            'metrics': [
                tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
                tfa.metrics.F1Score(average='macro', num_classes=4, name='macro_f1'),
            ],
            'classes': 4
        },
        'semeval14': {
            'train': 'train.xml',
            'val': 'test.xml',
            'test': 'test.xml',
            'unsupervised': [
                ('mams', 'train.xml'),
                ('semeval15', 'train.xml'),
                ('semeval16', 'train.xml')
            ],
            'metrics': [
                tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
                tfa.metrics.F1Score(average='macro', num_classes=5, name='macro_f1'),
            ],
            'classes': 5
        },
        'batch_size': 16,
        'epochs': 8,
        'optimizer': tf.keras.optimizers.Adam(learning_rate=1e-5),
        'loss': 'categorical_crossentropy',
        'result_path': 'results/raw',
        'baseline': {
            'result_path': 'results/raw'
        },
        'gan': {
            'result_path': 'results/raw'
        },
        'wgan': {
            'c_optimizer': tf.keras.optimizers.RMSprop(learning_rate=5e-5),
            'g_optimizer': tf.keras.optimizers.RMSprop(learning_rate=5e-5),
            'result_path': 'results/raw'
        },
    }


def baseline_experiment(dataset: str, fraction: float, name: str):
    tf.keras.backend.clear_session()

    config = get_config()
    data_config = config[dataset]
    model_config = config['baseline']

    # Load training, validation and testing data
    preprocessor = Preprocessor()
    (trainX, trainY), _ = preprocessor.parse_train(dataset, data_config['train'], validation_split=1 - fraction)
    val_data = preprocessor.parse_test(dataset, data_config['val'])
    test_data = preprocessor.parse_test(dataset, data_config['test'])

    # Load baseline model
    model = BaselineModel(data_config['classes'])
    model.compile(optimizer=config['optimizer'], loss=config['loss'], metrics=data_config['metrics'])

    return model.fit(
        trainX,
        trainY,
        batch_size=config['batch_size'],
        epochs=config['epochs'],
        validation_data=val_data,
        callbacks=[
            EvaluateCallback(test_data),
            ModelCheckpoint(os.path.join(config['result_path'], 'checkpoints', name),
                            save_best_only=True,
                            save_weights_only=True,
                            monitor='val_macro_f1',
                            mode='max'),
            CSVLogger(os.path.join(config['result_path'], f'{name}.csv'))
        ])


def baseline_experiments(datasets: List[str], fractions: List[float], runs: Iterator[int]):
    hists = []
    for dataset in datasets:
        for fraction in fractions:
            for run in runs:
                name = f'baseline_{dataset}_fr{fraction}_{run}'
                print(name)
                hists.append(baseline_experiment(dataset, fraction, name))

    return hists


# noinspection DuplicatedCode
def gan_experiment(dataset: str, fraction: float, unlabeled_ratio: float, name: str, generator=None, **callback_kwargs):
    tf.keras.backend.clear_session()

    config = get_config()
    data_config = config[dataset]
    model_config = config['gan']

    # Load training, validation and testing data
    preprocessor = Preprocessor()
    (trainX, trainY), _ = preprocessor.parse_train(
        dataset,
        data_config['train'],
        validation_split=1 - fraction,
        unlabeled_ratio=unlabeled_ratio,
        unlabeled_data=data_config['unsupervised']
    )
    val_data = preprocessor.parse_test(dataset, data_config['val'])
    test_data = preprocessor.parse_test(dataset, data_config['test'])

    # Load model
    model = BaseGAN(data_config['classes'], generator=generator)
    model.compile(optimizer=config['optimizer'], loss=config['loss'], metrics=data_config['metrics'])

    return model.fit(trainX, trainY, batch_size=config['batch_size'], epochs=config['epochs'],
                     validation_data=val_data,
                     callbacks=[
                         GANCallback(trainX, batch_size=config['batch_size'], **callback_kwargs),
                         EvaluateCallback(test_data),
                         ModelCheckpoint(
                             os.path.join(config['result_path'], 'checkpoints', name),
                             save_best_only=True,
                             save_weights_only=True,
                             monitor='val_macro_f1',
                             mode='max'),
                         CSVLogger(os.path.join(config['result_path'], f'{name}.csv'))
                     ])


def gan_experiments(datasets: List[str], fractions: List[float], unlabeled_ratios: List[float], runs: Iterator[int], generator=None, **callback_kwargs):
    hists = []
    for dataset in datasets:
        for fraction in fractions:
            for unlabeled_ratio in unlabeled_ratios:
                for run in runs:
                    name = f'gan_{dataset}_fr{fraction}_ur{unlabeled_ratio}_{run}'
                    print(name)
                    hists.append(gan_experiment(dataset, fraction, unlabeled_ratio, name, generator=generator, **callback_kwargs))

    return hists


# noinspection DuplicatedCode
def wgan_experiment(dataset: str, fraction: float, unlabeled_ratio: float, epb: int, name: str, generator=None):
    tf.keras.backend.clear_session()

    config = get_config()
    data_config = config[dataset]
    model_config = config['wgan']

    # Load training, validation and testing data
    preprocessor = Preprocessor()
    (trainX, trainY), _ = preprocessor.parse_train(
        dataset,
        data_config['train'],
        validation_split=1 - fraction,
        unlabeled_ratio=unlabeled_ratio,
        unlabeled_data=data_config['unsupervised']
    )
    val_data = preprocessor.parse_test(dataset, data_config['val'])
    test_data = preprocessor.parse_test(dataset, data_config['test'])

    # Load baseline model
    model = BaseGAN(data_config['classes'], generator=generator)
    model.compile(optimizer=config['optimizer'], loss=config['loss'], metrics=data_config['metrics'])

    wgan = WGAN(critic_steps=5, generator=model.generator)
    wgan.compile(model_config['c_optimizer'], model_config['g_optimizer'])

    return model.fit(trainX, trainY, batch_size=config['batch_size'], epochs=config['epochs'],
                     validation_data=val_data,
                     callbacks=[
                         WGANCallback(wgan, trainX, batch_size=config['batch_size'], epochs_per_batch=epb),
                         EvaluateCallback(test_data),
                         ModelCheckpoint(
                             os.path.join(config['result_path'], 'checkpoints', name),
                             save_best_only=True,
                             save_weights_only=True,
                             monitor='val_macro_f1',
                             mode='max'),
                         CSVLogger(os.path.join(config['result_path'], f'{name}.csv'))
                     ])


def wgan_experiments(datasets: List[str], fractions: List[float], unlabeled_ratios: List[float], epbs: List[int],
                     runs: Iterator[int], generator=None):
    hists = []
    for dataset in datasets:
        for fraction in fractions:
            for unlabeled_ratio in unlabeled_ratios:
                for epb in epbs:
                    for run in runs:
                        name = f'wgan_{dataset}_fr{fraction}_ur{unlabeled_ratio}_epb{epb}_{run}'
                        print(name)
                        hists.append(wgan_experiment(dataset, fraction, unlabeled_ratio, epb, name, generator=generator))

    return hists
