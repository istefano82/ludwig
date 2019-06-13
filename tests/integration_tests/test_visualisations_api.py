# -*- coding: utf-8 -*-
# Copyright (c) 2019 Uber Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import shutil
import glob
import pandas as pd

from ludwig.api import LudwigModel
from ludwig.utils.data_utils import read_csv
from ludwig import visualize
from tests.integration_tests.utils import ENCODERS
from tests.integration_tests.utils import categorical_feature
from tests.integration_tests.utils import generate_data
from tests.integration_tests.utils import sequence_feature
from tests.integration_tests.utils import text_feature
from unittest import mock

# The following imports are pytest fixtures, required for running the tests
from tests.fixtures.filenames import csv_filename


def run_api_experiment(input_features, output_features):
    """
    Helper method to avoid code repetition in running an experiment
    :param input_features: input schema
    :param output_features: output schema
    :param data_csv: path to data
    :return: None
    """
    model_definition = {
        'input_features': input_features,
        'output_features': output_features,
        'combiner': {'type': 'concat', 'fc_size': 14},
        'training': {'epochs': 2}
    }

    model = LudwigModel(model_definition)
    return model

def test_learning_curves_vis_api(csv_filename):
    """Ensure pdf and png figures can be saved via visualisation API call.

    :param csv_filename: csv fixture from tests.fixtures.filenames.csv_filename
    :return: None
    """
    # Single sequence input, single category output
    input_features = [sequence_feature(reduce_output='sum')]
    output_features = [categorical_feature(vocab_size=2, reduce_input='sum')]
    encoder = 'cnnrnn'

    # Generate test data
    data_csv = generate_data(input_features, output_features, csv_filename)
    input_features[0]['encoder'] = encoder
    model = run_api_experiment(input_features, output_features)
    data_df = read_csv(data_csv)
    train_stats = model.train(
        data_df=data_df,
        skip_save_processed_input=True,
        skip_save_progress=True,
        skip_save_unprocessed_output=True
    )
    viz_outputs = ('pdf', 'png')
    for viz_output in viz_outputs:
        vis_output_pattern_pdf = model.exp_dir_name + '/*.{}'.format(viz_output)
        visualize.learning_curves(
            train_stats,
            field=None,
            output_directory=model.exp_dir_name,
            file_format=viz_output
        )
        figure_cnt = glob.glob(vis_output_pattern_pdf)
        assert 5 == len(figure_cnt)
    model.close()
    shutil.rmtree(model.exp_dir_name, ignore_errors=True)


def test_compare_performance_vis_api(csv_filename):
    """Ensure pdf and png figures can be saved via visualisation API call.

    :param csv_filename: csv fixture from tests.fixtures.filenames.csv_filename
    :return: None
    """
    # Single sequence input, single category output
    input_features = [sequence_feature(reduce_output='sum')]
    output_features = [categorical_feature(vocab_size=2, reduce_input='sum')]
    encoder = 'cnnrnn'

    # Generate test data
    data_csv = generate_data(input_features, output_features, csv_filename)
    input_features[0]['encoder'] = encoder
    model = run_api_experiment(input_features, output_features)
    data_df = read_csv(data_csv)
    model.train(
        data_df=data_df,
        skip_save_processed_input=True,
        skip_save_progress=True,
        skip_save_unprocessed_output=True
    )
    test_stats = model.test(
        data_df=data_df
    )[1]
    viz_outputs = ('pdf', 'png')
    for viz_output in viz_outputs:
        vis_output_pattern_pdf = model.exp_dir_name + '/*.{}'.format(viz_output)
        visualize.compare_performance(
            [test_stats, test_stats],
            field=None,
            model_name = ['Model1', 'Model2'],
            output_directory=model.exp_dir_name,
            file_format=viz_output
        )
        figure_cnt = glob.glob(vis_output_pattern_pdf)
        assert 2 == len(figure_cnt)
    model.close()
    shutil.rmtree(model.exp_dir_name, ignore_errors=True)


def test_compare_classifier_performance_from_prob_vis_api(csv_filename):
    """Ensure pdf and png figures can be saved via visualisation API call.

    :param csv_filename: csv fixture from tests.fixtures.filenames.csv_filename
    :return: None
    """
    # Single sequence input, single category output
    input_features = [
        text_feature(vocab_size=10, min_len=1, representation='sparse'),
        categorical_feature(
            vocab_size=10,
            loss='sampled_softmax_cross_entropy'
        )
    ]
    output_features = [categorical_feature(vocab_size=2, reduce_input='sum')]
    encoder = 'cnnrnn'

    # Generate test data
    data_csv = generate_data(input_features, output_features, csv_filename)
    input_features[0]['encoder'] = encoder
    model = run_api_experiment(input_features, output_features)
    data_df = read_csv(data_csv)
    model.train(
        data_df=data_df,
        skip_save_processed_input=True,
        skip_save_progress=True,
        skip_save_unprocessed_output=True
    )
    test_stats = model.test(
        data_df=data_df
    )
    # probabilities need  to be list of lists containing each row data from the
    # probability columns
    probability = test_stats[0].iloc[:, 2:].values
    viz_outputs = ('pdf', 'png')
    field = output_features[0]['name']
    ground_truth = data_df[output_features[0]['name']]
    for viz_output in viz_outputs:
        vis_output_pattern_pdf = model.exp_dir_name + '/*.{}'.format(viz_output)
        visualize.compare_classifiers_performance_from_prob(
            [probability, probability],
            ground_truth,
            top_n_classes=[0],
            labels_limit=0,
            model_name = ['Model1', 'Model2'],
            output_directory=model.exp_dir_name,
            file_format=viz_output
        )
        figure_cnt = glob.glob(vis_output_pattern_pdf)
        assert 1 == len(figure_cnt)
    model.close()
    shutil.rmtree(model.exp_dir_name, ignore_errors=True)

def test_compare_classifier_performance_from_pred_vis_api(csv_filename):
    """Ensure pdf and png figures can be saved via visualisation API call.

    :param csv_filename: csv fixture from tests.fixtures.filenames.csv_filename
    :return: None
    """
    # Single sequence input, single category output
    input_features = [
        text_feature(vocab_size=10, min_len=1, representation='sparse'),
        categorical_feature(
            vocab_size=10,
            loss='sampled_softmax_cross_entropy'
        )
    ]
    output_features = [categorical_feature(vocab_size=2, reduce_input='sum')]
    encoder = 'cnnrnn'

    # Generate test data
    data_csv = generate_data(input_features, output_features, csv_filename)
    input_features[0]['encoder'] = encoder
    model = run_api_experiment(input_features, output_features)
    data_df = read_csv(data_csv)
    model.train(
        data_df=data_df,
        skip_save_processed_input=False,
        skip_save_progress=False,
        skip_save_unprocessed_output=False
    )
    test_stats = model.test(
        data_df=data_df
    )
    # predictions need  to be list of lists containing each row data from the
    # prediction column
    prediction = test_stats[0].iloc[:, 0].tolist()
    viz_outputs = ('pdf', 'png')
    field = output_features[0]['name']
    ground_truth = data_df[output_features[0]['name']]
    ground_truth_metadata = model.train_set_metadata
    for viz_output in viz_outputs:
        vis_output_pattern_pdf = model.exp_dir_name + '/*.{}'.format(viz_output)
        visualize.compare_classifiers_performance_from_pred(
            [prediction, prediction],
            ground_truth,
            ground_truth_metadata,
            field,
            labels_limit=0,
            model_name = ['Model1', 'Model2'],
            output_directory=model.exp_dir_name,
            file_format=viz_output
        )
        figure_cnt = glob.glob(vis_output_pattern_pdf)
        assert 1 == len(figure_cnt)
    model.close()
    # shutil.rmtree(model.exp_dir_name, ignore_errors=True)
