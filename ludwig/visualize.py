#! /usr/bin/env python
# coding=utf-8
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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import os
import sys
from functools import partial

import numpy as np
import sklearn
from scipy.stats import entropy
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

from ludwig.constants import *
from ludwig.contrib import contrib_command
from ludwig.utils import visualization_utils
from ludwig.utils.data_utils import load_json, load_from_file
from ludwig.utils.print_utils import logging_level_registry


def validate_conf_treshholds_and_probabilities_2d_3d(
        probabilities, treshhold_fields
):
    """Ensure probabilities and treshhold fields arrays have two members each.

    :param probabilities: List of probabilities per model
    :param treshhold_fields: List of treshhold fields per model
    :raise: RuntimeError
    """
    validation_mapping = {
        'probabilities': probabilities,
        'treshhold_fields': treshhold_fields
    }
    for item, value in validation_mapping.items():
        item_len = len(value)
        if not item_len == 2:
            exception_message = 'Two {} should be provided - {} was given.'.format(
                item,
                item_len
            )
            logging.error(exception_message)
            raise RuntimeError(exception_message)

def load_data_for_viz(load_type, model_file_statistics, *args, **kwargs):
    """Load model file data in to list of .

    :param load_type: type of the data loader to be used.
    :param model_file_statistics: JSON file or list of json files containing any
           model experiment stats.
    :return List of training statistics loaded as json objects.
    """
    SUPPORTED_LOAD_TYPES = dict(load_json=load_json,
                                load_from_file=partial(load_from_file,
                                                       dtype=kwargs.get('dtype',
                                                                        None)))
    loader = SUPPORTED_LOAD_TYPES[load_type]
    try:
        stats_per_model = [loader(stats_f)
                                for stats_f in
                                model_file_statistics]
    except (TypeError, AttributeError):
        logging.exception(
            'Unable to open model statistics file {}!'.format(
                model_file_statistics
            )
        )
        return
    return stats_per_model

def convert_to_list(item):
    """If item is not list class instance or None put inside a list.

    :param item: object to be checked and converted
    :return: original item if it is a list instance or list containing the item.
    """
    return item if item is None or isinstance(item, list) else [item]

def validate_visualisation_prediction_field_from_train_stats(
        field,
        train_stats_per_model
):
    """Validate prediction field from model train stats and return it as list.

    :param field: field containing ground truth
    :param train_stats_per_model: list of per model train stats
    :return fields: list of field(s) containing ground truth
    """
    fields_set = set()
    for ls in train_stats_per_model:
        for _, values in ls.items():
            for key in values:
                fields_set.add(key)
    try:
        return [field] if field in fields_set else fields_set
    except TypeError:
        return fields_set

def validate_visualisation_prediction_field_from_test_stats(
        field,
        test_stats_per_model
):
    """Validate prediction field from model test stats and return it as list.

    :param field: field containing ground truth
    :param test_stats_per_model: list of per model test stats
    :return fields: list of field(s) containing ground truth
    """
    fields_set = set()
    for ls in test_stats_per_model:
        for key in ls:
            fields_set.add(key)
    try:
        return [field] if field in fields_set else fields_set
    except TypeError:
        return fields_set

def generate_filename_template_path(output_dir, filename_template):
    """Ensure path to template file can be constructed given an output dir.

    Create output directory if yet does exist.
    :param output_dir: Directory that will contain the filename_template file
    :param filename_template: name of the file template to be appended to the
            filename template path
    :return: path to filename template inside the output dir or None if the
             output dir is None
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        return os.path.join(output_dir, filename_template)
    return None

def learning_curves(
        train_stats_per_model,
        field,
        model_names=None,
        output_directory=None,
        file_format='pdf',
        **kwargs
):
    filename_template = 'learning_curves_{}_{}.' + file_format
    filename_template_path = generate_filename_template_path(
        output_directory,
        filename_template
    )
    train_stats_per_model_list = convert_to_list(train_stats_per_model)
    model_names_list = convert_to_list(model_names)
    fields = validate_visualisation_prediction_field_from_train_stats(
        field,
        train_stats_per_model_list
    )

    metrics = [LOSS, ACCURACY, HITS_AT_K, EDIT_DISTANCE]
    for field in fields:
        for metric in metrics:
            if metric in train_stats_per_model_list[0]['train'][field]:
                filename = None
                if filename_template_path:
                    filename = filename_template_path.format(field, metric)

                visualization_utils.learning_curves_plot(
                    [learning_stats['train'][field][metric]
                     for learning_stats in train_stats_per_model_list],
                    [learning_stats['validation'][field][metric]
                     for learning_stats in train_stats_per_model_list],
                    metric,
                    model_names_list,
                    title='Learning Curves {}'.format(field),
                    filename=filename
                )

def compare_performance(
        test_stats_per_model,
        field, model_names=None,
        output_directory=None,
        file_format='pdf',
        **kwargs
):
    filename_template = 'compare_performance_{}.' + file_format
    filename_template_path = generate_filename_template_path(
        output_directory,
        filename_template
    )

    test_stats_per_model_list = convert_to_list(test_stats_per_model)
    model_names_list = convert_to_list(model_names)
    fields = validate_visualisation_prediction_field_from_test_stats(
        field,
        test_stats_per_model_list
    )

    for field in fields:
        accuracies = []
        hits_at_ks = []
        edit_distances = []

        for test_stats_per_model in test_stats_per_model_list:
            if ACCURACY in test_stats_per_model[field]:
                accuracies.append(test_stats_per_model[field][ACCURACY])
            if HITS_AT_K in test_stats_per_model[field]:
                hits_at_ks.append(test_stats_per_model[field][HITS_AT_K])
            if EDIT_DISTANCE in test_stats_per_model[field]:
                edit_distances.append(
                    test_stats_per_model[field][EDIT_DISTANCE])

        measures = []
        measures_names = []
        if len(accuracies) > 0:
            measures.append(accuracies)
            measures_names.append(ACCURACY)
        if len(hits_at_ks) > 0:
            measures.append(hits_at_ks)
            measures_names.append(HITS_AT_K)
        if len(edit_distances) > 0:
            measures.append(edit_distances)
            measures_names.append(EDIT_DISTANCE)

        filename = None
        if filename_template_path:
            filename = filename_template_path.format(field)
            os.makedirs(output_directory, exist_ok=True)

        visualization_utils.compare_classifiers_plot(
            measures,
            measures_names,
            model_names_list,
            title='Performance comparison on {}'.format(field),
            filename=filename
        )


def compare_classifiers_performance_from_prob(
        probs_per_model,
        gt,
        top_n_classes,
        labels_limit,
        model_names=None,
        output_directory=None,
        file_format='pdf',
        **kwargs
):
    k = top_n_classes[0]
    model_names_list = convert_to_list(model_names)
    if labels_limit > 0:
        gt[gt > labels_limit] = labels_limit

    probs = probs_per_model
    accuracies = []
    hits_at_ks = []
    mrrs = []

    for i, prob in enumerate(probs):

        if labels_limit > 0 and prob.shape[1] > labels_limit + 1:
            prob_limit = prob[:, :labels_limit + 1]
            prob_limit[:, labels_limit] = prob[:, labels_limit:].sum(1)
            prob = prob_limit

        prob = np.argsort(prob, axis=1)
        top1 = prob[:, -1]
        topk = prob[:, -k:]

        accuracies.append((gt == top1).sum() / len(gt))

        hits_at_k = 0
        for j in range(len(gt)):
            hits_at_k += np.in1d(gt[j], topk[i, :])
        hits_at_ks.append(np.asscalar(hits_at_k) / len(gt))

        mrr = 0
        for j in range(len(gt)):
            gt_pos_in_probs = prob[i, :] == gt[j]
            if np.any(gt_pos_in_probs):
                mrr += (1 / -(np.asscalar(np.argwhere(gt_pos_in_probs)) -
                              prob.shape[1]))
        mrrs.append(mrr / len(gt))

    filename = None
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)
        filename = os.path.join(
            output_directory,
            'compare_classifiers_performance_from_prob.' + file_format
        )

    visualization_utils.compare_classifiers_plot(
        [accuracies, hits_at_ks, mrrs],
        [ACCURACY, HITS_AT_K, 'mrr'],
        model_names_list,
        filename=filename
    )


def compare_classifiers_performance_from_pred(
        preds_per_model,
        gt,
        metadata,
        field,
        labels_limit,
        model_names=None,
        output_directory=None,
        file_format='pdf',
        **kwargs
):

    if labels_limit > 0:
        gt[gt > labels_limit] = labels_limit

    preds = preds_per_model
    model_names_list = convert_to_list(model_names)
    mapped_preds = []
    for pred in preds:
        mapped_preds.append([metadata[field]['str2idx'][val] for val in pred])
    preds = mapped_preds
    accuracies = []
    precisions = []
    recalls = []
    f1s = []

    for i, pred in enumerate(preds):
        accuracies.append(sklearn.metrics.accuracy_score(gt, pred))
        precisions.append(
            sklearn.metrics.precision_score(gt, pred, average='macro')
        )
        recalls.append(sklearn.metrics.recall_score(gt, pred, average='macro'))
        f1s.append(sklearn.metrics.f1_score(gt, pred, average='macro'))

    filename = None
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)
        filename = os.path.join(
            output_directory,
            'compare_classifiers_performance_from_pred.' + file_format
        )

    visualization_utils.compare_classifiers_plot(
        [accuracies, precisions, recalls, f1s],
        [ACCURACY, 'precision', 'recall', 'f1'],
        model_names_list,
        filename=filename
    )


def compare_classifiers_performance_subset(
        probs_per_model,
        gt,
        top_n_classes,
        labels_limit,
        subset,
        model_names=None,
        output_directory=None,
        file_format='pdf',
        **kwargs
):
    k = top_n_classes[0]
    model_names_list = convert_to_list(model_names)
    if labels_limit > 0:
        gt[gt > labels_limit] = labels_limit

    subset_indices = gt > 0
    gt_subset = gt
    if subset == 'ground_truth':
        subset_indices = gt < k
        gt_subset = gt[subset_indices]
        logging.info('Subset is {:.2f}% of the data'.format(
            len(gt_subset) / len(gt) * 100)
        )

    probs = probs_per_model
    accuracies = []
    hits_at_ks = []

    for i, prob in enumerate(probs):

        if labels_limit > 0 and prob.shape[1] > labels_limit + 1:
            prob_limit = prob[:, :labels_limit + 1]
            prob_limit[:, labels_limit] = prob[:, labels_limit:].sum(1)
            prob = prob_limit

        if subset == PREDICTIONS:
            subset_indices = np.argmax(prob, axis=1) < k
            gt_subset = gt[subset_indices]
            logging.info(
                'Subset for model_name {} is {:.2f}% of the data'.format(
                    model_names[i] if model_names and i < len(
                        model_names) else i,
                    len(gt_subset) / len(gt) * 100
                )
            )
            model_names[i] = '{} ({:.2f}%)'.format(
                model_names[i] if model_names and i < len(model_names) else i,
                len(gt_subset) / len(gt) * 100
            )

        prob_subset = prob[subset_indices]

        prob_subset = np.argsort(prob_subset, axis=1)
        top1_subset = prob_subset[:, -1]
        top3_subset = prob_subset[:, -3:]

        accuracies.append(np.sum((gt_subset == top1_subset)) / len(gt_subset))

        hits_at_k = 0
        for j in range(len(gt_subset)):
            hits_at_k += np.in1d(gt_subset[j], top3_subset[i, :])
        hits_at_ks.append(np.asscalar(hits_at_k) / len(gt_subset))

    title = None
    if subset == 'ground_truth':
        title = 'Classifier performance on first {} class{} ({:.2f}%)'.format(
            k, 'es' if k > 1 else '', len(gt_subset) / len(gt) * 100
        )
    elif subset == PREDICTIONS:
        title = 'Classifier performance on first {} class{}'.format(
            k,
            'es' if k > 1 else ''
        )

    filename = None
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)
        filename = os.path.join(
            output_directory,
            'compare_classifiers_performance_subset.' + file_format
        )

    visualization_utils.compare_classifiers_plot(
        [accuracies, hits_at_ks],
        [ACCURACY, HITS_AT_K],
        model_names_list,
        title=title,
        filename=filename
    )


def compare_classifiers_performance_changing_k(
        probs_per_model,
        gt,
        top_k,
        labels_limit,
        model_names=None,
        output_directory=None,
        file_format='pdf',
        **kwargs
):
    k = top_k
    if labels_limit > 0:
        gt[gt > labels_limit] = labels_limit
    probs = probs_per_model

    hits_at_ks = []
    model_names_list = convert_to_list(model_names)
    for i, prob in enumerate(probs):

        if labels_limit > 0 and prob.shape[1] > labels_limit + 1:
            prob_limit = prob[:, :labels_limit + 1]
            prob_limit[:, labels_limit] = prob[:, labels_limit:].sum(1)
            prob = prob_limit

        prob = np.argsort(prob, axis=1)

        hits_at_k = [0.0] * k
        for g in range(len(gt)):
            for j in range(k):
                hits_at_k[j] += np.in1d(gt[g], prob[g, -j - 1:])
        hits_at_ks.append(np.array(hits_at_k) / len(gt))

    filename = None
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)
        filename = os.path.join(
            output_directory,
            'compare_classifiers_performance_changing_k.' + file_format
        )

    visualization_utils.compare_classifiers_line_plot(
        np.arange(1, k + 1),
        hits_at_ks, 'hits@k',
        model_names_list,
        title='Classifier comparison (hits@k)',
        filename=filename
    )


def compare_classifiers_multiclass_multimetric(
        test_stats_per_model,
        metadata,
        field,
        top_n_classes,
        model_names=None,
        output_directory=None,
        file_format='pdf',
        **kwargs
):
    filename_template = 'compare_classifiers_multiclass_multimetric_{}_{}_{}.' \
                        + file_format
    filename_template_path = generate_filename_template_path(
        output_directory,
        filename_template
    )

    test_stats_per_model_list = convert_to_list(test_stats_per_model)
    model_names_list = convert_to_list(model_names)
    fields = validate_visualisation_prediction_field_from_test_stats(
        field,
        test_stats_per_model_list
    )

    for i, test_statistics in enumerate(
            test_stats_per_model_list):
        for field in fields:
            model_name_name = (
                model_names_list[i]
                if model_names_list is not None and i < len(model_names_list)
                else ''
            )
            if 'per_class_stats' not in test_statistics[field]:
                logging.warning(
                    'The field {} in test statistics does not contain "{}", '
                    'skipping it'.format(field, per_class_stats)
                )
                break
            per_class_stats = test_statistics[field]['per_class_stats']
            precisions = []
            recalls = []
            f1_scores = []
            labels = []
            for _, class_name in sorted(
                    [(metadata[field]['str2idx'][key], key)
                     for key in per_class_stats.keys()],
                    key=lambda tup: tup[0]):
                class_stats = per_class_stats[class_name]
                precisions.append(class_stats['precision'])
                recalls.append(class_stats['recall'])
                f1_scores.append(class_stats['f1_score'])
                labels.append(class_name)
            for k in top_n_classes:
                k = min(k, len(precisions)) if k > 0 else len(precisions)
                ps = precisions[0:k]
                rs = recalls[0:k]
                fs = f1_scores[0:k]
                ls = labels[0:k]

                filename = None
                if filename_template_path:
                    os.makedirs(output_directory, exist_ok=True)
                    filename = filename_template_path.format(
                        model_name_name, field, 'top{}'.format(k)
                    )

                visualization_utils.compare_classifiers_multiclass_multimetric_plot(
                    [ps, rs, fs],
                    ['precision', 'recall', 'f1 score'],
                    labels=ls,
                    title='{} Multiclass Precision / Recall / '
                          'F1 Score top {} {}'.format(model_name_name, k,
                                                      field),
                    filename=filename
                )

                p_np = np.nan_to_num(np.array(precisions, dtype=np.float32))
                r_np = np.nan_to_num(np.array(recalls, dtype=np.float32))
                f1_np = np.nan_to_num(np.array(f1_scores, dtype=np.float32))
                labels_np = np.nan_to_num(np.array(labels))

                sorted_indices = f1_np.argsort()
                higher_f1s = sorted_indices[-k:][::-1]
                filename = None
                if filename_template_path:
                    os.makedirs(output_directory, exist_ok=True)
                    filename = filename_template_path.format(
                        model_name_name, field, 'best{}'.format(k)
                    )
                visualization_utils.compare_classifiers_multiclass_multimetric_plot(
                    [p_np[higher_f1s],
                     r_np[higher_f1s],
                     f1_np[higher_f1s]],
                    ['precision', 'recall', 'f1 score'],
                    labels=labels_np[higher_f1s].tolist(),
                    title='{} Multiclass Precision / Recall / '
                          'F1 Score best {} classes {}'.format(
                        model_name_name, k, field),
                    filename=filename
                )
                lower_f1s = sorted_indices[:k]
                filename = None
                if filename_template_path:
                    filename = filename_template_path.format(
                        model_name_name, field, 'worst{}'.format(k)
                    )
                visualization_utils.compare_classifiers_multiclass_multimetric_plot(
                    [p_np[lower_f1s],
                     r_np[lower_f1s],
                     f1_np[lower_f1s]],
                    ['precision', 'recall', 'f1 score'],
                    labels=labels_np[lower_f1s].tolist(),
                    title='{} Multiclass Precision / Recall / F1 Score worst '
                          'k classes {}'.format(model_name_name, k, field),
                    filename=filename
                )

                filename = None
                if filename_template_path:
                    filename = filename_template_path.format(
                        model_name_name, field, 'sorted'
                    )
                visualization_utils.compare_classifiers_multiclass_multimetric_plot(
                    [p_np[sorted_indices[::-1]],
                     r_np[sorted_indices[::-1]],
                     f1_np[sorted_indices[::-1]]],
                    ['precision', 'recall', 'f1 score'],
                    labels=labels_np[sorted_indices[::-1]].tolist(),
                    title='{} Multiclass Precision / Recall / F1 Score '
                          '{} sorted'.format(model_name_name, field),
                    filename=filename
                )

                logging.info('\n')
                logging.info(model_name_name)
                tmp_str = '{0} best 5 classes: '.format(field)
                tmp_str += '{}'
                logging.info(tmp_str.format(higher_f1s))
                logging.info(f1_np[higher_f1s])
                tmp_str = '{0} worst 5 classes: '.format(field)
                tmp_str += '{}'
                logging.info(tmp_str.format(lower_f1s))
                logging.info(f1_np[lower_f1s])
                tmp_str = '{0} number of classes with f1 score > 0: '.format(
                    field)
                tmp_str += '{}'
                logging.info(tmp_str.format(np.sum(f1_np > 0)))
                tmp_str = '{0} number of classes with f1 score = 0: '.format(
                    field)
                tmp_str += '{}'
                logging.info(tmp_str.format(np.sum(f1_np == 0)))


def compare_classifiers_predictions(
        preds_per_model,
        gt,
        labels_limit,
        model_names=None,
        output_directory=None,
        file_format='pdf',
        **kwargs
):
    model_names_list = convert_to_list(model_names)
    name_c1 = (
        model_names_list[0] if model_names is not None and len(model_names) > 0
        else 'c1')
    name_c2 = (
        model_names_list[1] if model_names is not None and len(model_names) > 1
        else 'c2')


    pred_c1 = preds_per_model[0]
    pred_c2 = preds_per_model[1]

    if labels_limit > 0:
        gt[gt > labels_limit] = labels_limit
        pred_c1[pred_c1 > labels_limit] = labels_limit
        pred_c2[pred_c2 > labels_limit] = labels_limit

    # DOTO all shadows built in name - come up with a more descriptive name
    all = len(gt)
    if all == 0:
        logging.error('No labels in the ground truth')
        return

    both_right = 0
    both_wrong_same = 0
    both_wrong_different = 0
    c1_right_c2_wrong = 0
    c1_wrong_c2_right = 0

    for i in range(all):
        if gt[i] == pred_c1[i] and gt[i] == pred_c2[i]:
            both_right += 1
        elif gt[i] != pred_c1[i] and gt[i] != pred_c2[i]:
            if pred_c1[i] == pred_c2[i]:
                both_wrong_same += 1
            else:
                both_wrong_different += 1
        elif gt[i] == pred_c1[i] and gt[i] != pred_c2[i]:
            c1_right_c2_wrong += 1
        elif gt[i] != pred_c1[i] and gt[i] == pred_c2[i]:
            c1_wrong_c2_right += 1

    one_right = c1_right_c2_wrong + c1_wrong_c2_right
    both_wrong = both_wrong_same + both_wrong_different

    logging.info('Test datapoints: {}'.format(all))
    logging.info(
        'Both right: {} {:.2f}%'.format(both_right, 100 * both_right / all))
    logging.info(
        'One right: {} {:.2f}%'.format(one_right, 100 * one_right / all))
    logging.info(
        '  {} right / {} wrong: {} {:.2f}% {:.2f}%'.format(
            name_c1,
            name_c2,
            c1_right_c2_wrong,
            100 * c1_right_c2_wrong / all,
            100 * c1_right_c2_wrong / one_right if one_right > 0 else 0
        )
    )
    logging.info(
        '  {} wrong / {} right: {} {:.2f}% {:.2f}%'.format(
            name_c1,
            name_c2,
            c1_wrong_c2_right,
            100 * c1_wrong_c2_right / all,
            100 * c1_wrong_c2_right / one_right if one_right > 0 else 0
        )
    )
    logging.info(
        'Both wrong: {} {:.2f}%'.format(both_wrong, 100 * both_wrong / all)
    )
    logging.info('  same prediction: {} {:.2f}% {:.2f}%'.format(
        both_wrong_same,
        100 * both_wrong_same / all,
        100 * both_wrong_same / both_wrong if both_wrong > 0 else 0
    )
    )
    logging.info('  different prediction: {} {:.2f}% {:.2f}%'.format(
        both_wrong_different,
        100 * both_wrong_different / all,
        100 * both_wrong_different / both_wrong if both_wrong > 0 else 0
    )
    )

    filename = None
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)
        filename = os.path.join(
            output_directory,
            'compare_classifiers_predictions_{}_{}.{}'.format(
                name_c1, name_c2, file_format
            )
        )

    visualization_utils.donut(
        [both_right, one_right, both_wrong],
        ['both right', 'one right', 'both wrong'],
        [both_right, c1_right_c2_wrong, c1_wrong_c2_right, both_wrong_same,
         both_wrong_different],
        ['both right',
         '{} right / {} wrong'.format(name_c1, name_c2),
         '{} wrong / {} right'.format(name_c1, name_c2),
         'same prediction', 'different prediction'],
        [0, 1, 1, 2, 2],
        title='{} vs {}'.format(name_c1, name_c2),
        filename=filename
    )

def compare_classifiers_predictions_distribution(
        preds_per_model,
        gt,
        labels_limit,
        model_names=None,
        output_directory=None,
        file_format='pdf',
        **kwargs
):
    model_names_list = convert_to_list(model_names)
    if labels_limit > 0:
        gt[gt > labels_limit] = labels_limit
        for i in range(len(preds_per_model)):
            preds_per_model[i][preds_per_model[i] > labels_limit] = labels_limit

    counts_gt = np.bincount(gt)
    prob_gt = counts_gt / counts_gt.sum()

    counts_predictions = [np.bincount(alg_predictions)
                          for alg_predictions in preds_per_model]

    prob_predictions = [alg_count_prediction / alg_count_prediction.sum()
                        for alg_count_prediction in counts_predictions]

    filename = None
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)
        filename = os.path.join(
            output_directory,
            'compare_classifiers_predictions_distribution.' + file_format
        )

    visualization_utils.radar_chart(
        prob_gt,
        prob_predictions,
        model_names_list,
        filename=filename
    )


def confidence_thresholding(
        probs_per_model,
        gt,
        labels_limit,
        model_names=None,
        output_directory=None,
        file_format='pdf',
        **kwargs
):
    if labels_limit > 0:
        gt[gt > labels_limit] = labels_limit
    probs = probs_per_model
    model_names_list = convert_to_list(model_names)
    thresholds = [t / 100 for t in range(0, 101, 5)]

    accuracies = []
    dataset_kept = []

    for i, prob in enumerate(probs):

        if labels_limit > 0 and prob.shape[1] > labels_limit + 1:
            prob_limit = prob[:, :labels_limit + 1]
            prob_limit[:, labels_limit] = prob[:, labels_limit:].sum(1)
            prob = prob_limit

        max_prob = np.max(prob, axis=1)
        predictions = np.argmax(prob, axis=1)

        accuracies_alg = []
        dataset_kept_alg = []

        for threshold in thresholds:
            threshold = threshold if threshold < 1 else 0.999
            filtered_indices = max_prob >= threshold
            filtered_gt = gt[filtered_indices]
            filtered_predictions = predictions[filtered_indices]
            accuracy = (
                    (filtered_gt == filtered_predictions).sum() /
                    len(filtered_gt)
            )

            accuracies_alg.append(accuracy)
            dataset_kept_alg.append(len(filtered_gt) / len(gt))

        accuracies.append(accuracies_alg)
        dataset_kept.append(dataset_kept_alg)

    filename = None
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)
        filename = os.path.join(
            output_directory,
            'confidence_thresholding.' + file_format
        )

    visualization_utils.confidence_fitlering_plot(
        thresholds,
        accuracies,
        dataset_kept,
        model_names_list,
        title='Confidence_Thresholding',
        filename=filename
    )


def confidence_thresholding_data_vs_acc(
        probs_per_model,
        gt,
        labels_limit,
        model_names=None,
        output_directory=None,
        file_format='pdf',
        **kwargs
):
    if labels_limit > 0:
        gt[gt > labels_limit] = labels_limit
    probs = probs_per_model
    model_names_list = convert_to_list(model_names)
    thresholds = [t / 100 for t in range(0, 101, 5)]

    accuracies = []
    dataset_kept = []

    for i, prob in enumerate(probs):

        if labels_limit > 0 and prob.shape[1] > labels_limit + 1:
            prob_limit = prob[:, :labels_limit + 1]
            prob_limit[:, labels_limit] = prob[:, labels_limit:].sum(1)
            prob = prob_limit

        max_prob = np.max(prob, axis=1)
        predictions = np.argmax(prob, axis=1)

        accuracies_alg = []
        dataset_kept_alg = []

        for threshold in thresholds:
            threshold = threshold if threshold < 1 else 0.999
            filtered_indices = max_prob >= threshold
            filtered_gt = gt[filtered_indices]
            filtered_predictions = predictions[filtered_indices]
            accuracy = ((filtered_gt == filtered_predictions).sum() /
                        len(filtered_gt))

            accuracies_alg.append(accuracy)
            dataset_kept_alg.append(len(filtered_gt) / len(gt))

        accuracies.append(accuracies_alg)
        dataset_kept.append(dataset_kept_alg)

    filename = None
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)
        filename = os.path.join(
            output_directory,
            'confidence_thresholding_data_vs_acc.' + file_format
        )

    visualization_utils.confidence_fitlering_data_vs_acc_plot(
        accuracies,
        dataset_kept,
        model_names_list,
        title='Confidence_Thresholding (Data vs Accuracy)',
        filename=filename
    )


def confidence_thresholding_data_vs_acc_subset(
        probs_per_model,
        gt,
        top_n_classes,
        labels_limit,
        subset,
        model_names=None,
        output_directory=None,
        file_format='pdf',
        **kwargs
):
    k = top_n_classes[0]
    if labels_limit > 0:
        gt[gt > labels_limit] = labels_limit
    probs = probs_per_model
    model_names_list = convert_to_list(model_names)
    thresholds = [t / 100 for t in range(0, 101, 5)]

    accuracies = []
    dataset_kept = []

    subset_indices = gt > 0
    gt_subset = gt
    if subset == 'ground_truth':
        subset_indices = gt < k
        gt_subset = gt[subset_indices]
        logging.info('Subset is {:.2f}% of the data'.format(
            len(gt_subset) / len(gt) * 100)
        )

    for i, prob in enumerate(probs):

        if labels_limit > 0 and prob.shape[1] > labels_limit + 1:
            prob_limit = prob[:, :labels_limit + 1]
            prob_limit[:, labels_limit] = prob[:, labels_limit:].sum(1)
            prob = prob_limit

        if subset == PREDICTIONS:
            subset_indices = np.argmax(prob, axis=1) < k
            gt_subset = gt[subset_indices]
            logging.info(
                'Subset for model_name {} is {:.2f}% of the data'.format(
                    model_names[i] if model_names and i < len(
                        model_names) else i,
                    len(gt_subset) / len(gt) * 100
                )
            )

        prob_subset = prob[subset_indices]

        max_prob = np.max(prob_subset, axis=1)
        predictions = np.argmax(prob_subset, axis=1)

        accuracies_alg = []
        dataset_kept_alg = []

        for threshold in thresholds:
            threshold = threshold if threshold < 1 else 0.999
            filtered_indices = max_prob >= threshold
            filtered_gt = gt_subset[filtered_indices]
            filtered_predictions = predictions[filtered_indices]
            accuracy = ((filtered_gt == filtered_predictions).sum() /
                        len(filtered_gt))

            accuracies_alg.append(accuracy)
            dataset_kept_alg.append(len(filtered_gt) / len(gt))

        accuracies.append(accuracies_alg)
        dataset_kept.append(dataset_kept_alg)

    filename = None
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)
        filename = os.path.join(
            output_directory,
            'confidence_thresholding_data_vs_acc_subset.' + file_format
        )

    visualization_utils.confidence_fitlering_data_vs_acc_plot(
        accuracies,
        dataset_kept,
        model_names_list,
        title='Confidence_Thresholding (Data vs Accuracy)',
        filename=filename
    )


def confidence_thresholding_data_vs_acc_subset_per_class(
        probs_per_model,
        gt,
        metadata,
        field,
        top_n_classes,
        labels_limit,
        subset,
        model_names=None,
        output_directory=None,
        file_format='pdf',
        **kwargs
):
    filename_template = 'confidence_thresholding_data_vs_acc_subset_per_class_{}.' + file_format
    filename_template_path = generate_filename_template_path(
        output_directory,
        filename_template
    )
    k = top_n_classes[0]
    if labels_limit > 0:
        gt[gt > labels_limit] = labels_limit
    probs = probs_per_model
    model_names_list = convert_to_list(model_names)

    thresholds = [t / 100 for t in range(0, 101, 5)]

    for curr_k in range(k):
        accuracies = []
        dataset_kept = []

        subset_indices = gt > 0
        gt_subset = gt
        if subset == 'ground_truth':
            subset_indices = gt == curr_k
            gt_subset = gt[subset_indices]
            logging.info('Subset is {:.2f}% of the data'.format(
                len(gt_subset) / len(gt) * 100)
            )

        for i, prob in enumerate(probs):

            if labels_limit > 0 and prob.shape[1] > labels_limit + 1:
                prob_limit = prob[:, :labels_limit + 1]
                prob_limit[:, labels_limit] = prob[:, labels_limit:].sum(1)
                prob = prob_limit

            if subset == PREDICTIONS:
                subset_indices = np.argmax(prob, axis=1) == curr_k
                gt_subset = gt[subset_indices]
                logging.info(
                    'Subset for model_name {} is {:.2f}% of the data'.format(
                        model_names_list[i] if model_names_list and i < len(
                            model_names_list) else i,
                        len(gt_subset) / len(gt) * 100
                    )
                )

            prob_subset = prob[subset_indices]

            max_prob = np.max(prob_subset, axis=1)
            predictions = np.argmax(prob_subset, axis=1)

            accuracies_alg = []
            dataset_kept_alg = []

            for threshold in thresholds:
                threshold = threshold if threshold < 1 else 0.999
                filtered_indices = max_prob >= threshold
                filtered_gt = gt_subset[filtered_indices]
                filtered_predictions = predictions[filtered_indices]
                accuracy = ((filtered_gt == filtered_predictions).sum() /
                            len(filtered_gt) if len(filtered_gt) > 0 else 0)

                accuracies_alg.append(accuracy)
                dataset_kept_alg.append(len(filtered_gt) / len(gt))

            accuracies.append(accuracies_alg)
            dataset_kept.append(dataset_kept_alg)

        field_name = metadata[field]['idx2str'][curr_k]

        filename = None
        if filename_template_path:
            os.makedirs(output_directory, exist_ok=True)
            filename = filename_template_path.format(field_name)

        visualization_utils.confidence_fitlering_data_vs_acc_plot(
            accuracies, dataset_kept, model_names_list,
            decimal_digits=2,
            title='Confidence_Thresholding (Data vs Accuracy) '
                  'for class {}'.format(field_name),
            filename=filename
        )


def confidence_thresholding_2thresholds_2d(
        probs_per_model,
        ground_truths,
        threshold_fields,
        labels_limit,
        model_names=None,
        output_directory=None,
        file_format='pdf',
        **kwargs
):
    try:
        validate_conf_treshholds_and_probabilities_2d_3d(
            probs_per_model,
            threshold_fields
        )
    except RuntimeError:
        return
    probs = probs_per_model
    model_names_list = convert_to_list(model_names)
    filename_template = 'confidence_thresholding_2thresholds_2d_{}.' + file_format
    filename_template_path = generate_filename_template_path(
        output_directory,
        filename_template
    )
    gt_1 = ground_truths[0]
    gt_2 = ground_truths[1]

    if labels_limit > 0:
        gt_1[gt_1 > labels_limit] = labels_limit
        gt_2[gt_2 > labels_limit] = labels_limit


    thresholds = [t / 100 for t in range(0, 101, 5)]
    fixed_step_coverage = thresholds
    name_t1 = '{} threshold'.format(threshold_fields[0])
    name_t2 = '{} threshold'.format(threshold_fields[1])

    accuracies = []
    dataset_kept = []
    interps = []
    table = [[name_t1, name_t2, 'coverage', ACCURACY]]

    if labels_limit > 0 and probs[0].shape[1] > labels_limit + 1:
        prob_limit = probs[0][:, :labels_limit + 1]
        prob_limit[:, labels_limit] = probs[0][:, labels_limit:].sum(1)
        probs[0] = prob_limit

    if labels_limit > 0 and probs[1].shape[1] > labels_limit + 1:
        prob_limit = probs[1][:, :labels_limit + 1]
        prob_limit[:, labels_limit] = probs[1][:, labels_limit:].sum(1)
        probs[1] = prob_limit

    max_prob_1 = np.max(probs[0], axis=1)
    predictions_1 = np.argmax(probs[0], axis=1)

    max_prob_2 = np.max(probs[1], axis=1)
    predictions_2 = np.argmax(probs[1], axis=1)

    for threshold_1 in thresholds:
        threshold_1 = threshold_1 if threshold_1 < 1 else 0.999
        curr_accuracies = []
        curr_dataset_kept = []

        for threshold_2 in thresholds:
            threshold_2 = threshold_2 if threshold_2 < 1 else 0.999

            filtered_indices = np.logical_and(
                max_prob_1 >= threshold_1,
                max_prob_2 >= threshold_2
            )

            filtered_gt_1 = gt_1[filtered_indices]
            filtered_predictions_1 = predictions_1[filtered_indices]
            filtered_gt_2 = gt_2[filtered_indices]
            filtered_predictions_2 = predictions_2[filtered_indices]

            coverage = len(filtered_gt_1) / len(gt_1)
            accuracy = (
                           np.logical_and(
                               filtered_gt_1 == filtered_predictions_1,
                               filtered_gt_2 == filtered_predictions_2
                           )
                       ).sum() / len(filtered_gt_1)

            curr_accuracies.append(accuracy)
            curr_dataset_kept.append(coverage)
            table.append([threshold_1, threshold_2, coverage, accuracy])

        accuracies.append(curr_accuracies)
        dataset_kept.append(curr_dataset_kept)
        interps.append(
            np.interp(
                fixed_step_coverage,
                list(reversed(curr_dataset_kept)),
                list(reversed(curr_accuracies)),
                left=1,
                right=0
            )
        )

    logging.info('CSV table')
    for row in table:
        logging.info(','.join([str(e) for e in row]))

    # ===========#
    # Multiline #
    # ===========#
    filename = None
    if filename_template_path:
        os.makedirs(output_directory, exist_ok=True)
        filename = filename_template_path.format('multiline')
    visualization_utils.confidence_fitlering_data_vs_acc_multiline_plot(
        accuracies,
        dataset_kept,
        model_names_list,
        title='Coverage vs Accuracy, two thresholds',
        filename=filename
    )

    # ==========#
    # Max line #
    # ==========#
    filename = None
    if filename_template_path:
        filename = filename_template_path.format('maxline')
    max_accuracies = np.amax(np.array(interps), 0)
    visualization_utils.confidence_fitlering_data_vs_acc_plot(
        [max_accuracies],
        [thresholds],
        model_names_list,
        title='Coverage vs Accuracy, two thresholds',
        filename=filename
    )

    # ==========================#
    # Max line with thresholds #
    # ==========================#
    acc_matrix = np.array(accuracies)
    cov_matrix = np.array(dataset_kept)
    t1_maxes = [1]
    t2_maxes = [1]
    for i in range(len(fixed_step_coverage) - 1):
        lower = fixed_step_coverage[i]
        upper = fixed_step_coverage[i + 1]
        indices = np.logical_and(cov_matrix >= lower, cov_matrix < upper)
        selected_acc = acc_matrix.copy()
        selected_acc[np.logical_not(indices)] = -1
        threshold_indices = np.unravel_index(np.argmax(selected_acc, axis=None),
                                             selected_acc.shape)
        t1_maxes.append(thresholds[threshold_indices[0]])
        t2_maxes.append(thresholds[threshold_indices[1]])
    model_name = model_names_list[0] if model_names_list is not None and len(
        model_names_list) > 0 else ''

    filename = None
    if filename_template_path:
        os.makedirs(output_directory, exist_ok=True)
        filename = filename_template_path.format('maxline_with_thresholds')

    visualization_utils.confidence_fitlering_data_vs_acc_plot(
        [max_accuracies, t1_maxes, t2_maxes],
        [fixed_step_coverage, fixed_step_coverage, fixed_step_coverage],
        model_names=[model_name + ' accuracy', name_t1, name_t2],
        dotted=[False, True, True],
        y_label='',
        title='Coverage vs Accuracy & Threshold',
        filename=filename
    )


def confidence_thresholding_2thresholds_3d(
        probs_per_model,
        ground_truths,
        threshold_fields,
        labels_limit,
        output_directory=None,
        file_format='pdf',
        **kwargs
):
    try:
        validate_conf_treshholds_and_probabilities_2d_3d(
            probs_per_model,
            threshold_fields
        )
    except RuntimeError:
        return
    probs = probs_per_model
    gt_1 = ground_truths[0]
    gt_2 = ground_truths[1]


    if labels_limit > 0:
        gt_1[gt_1 > labels_limit] = labels_limit
        gt_2[gt_2 > labels_limit] = labels_limit

    thresholds = [t / 100 for t in range(0, 101, 5)]

    accuracies = []
    dataset_kept = []

    if labels_limit > 0 and probs[0].shape[1] > labels_limit + 1:
        prob_limit = probs[0][:, :labels_limit + 1]
        prob_limit[:, labels_limit] = probs[0][:, labels_limit:].sum(1)
        probs[0] = prob_limit

    if labels_limit > 0 and probs[1].shape[1] > labels_limit + 1:
        prob_limit = probs[1][:, :labels_limit + 1]
        prob_limit[:, labels_limit] = probs[1][:, labels_limit:].sum(1)
        probs[1] = prob_limit

    max_prob_1 = np.max(probs[0], axis=1)
    predictions_1 = np.argmax(probs[0], axis=1)

    max_prob_2 = np.max(probs[1], axis=1)
    predictions_2 = np.argmax(probs[1], axis=1)

    for threshold_1 in thresholds:
        threshold_1 = threshold_1 if threshold_1 < 1 else 0.999
        curr_accuracies = []
        curr_dataset_kept = []

        for threshold_2 in thresholds:
            threshold_2 = threshold_2 if threshold_2 < 1 else 0.999

            filtered_indices = np.logical_and(
                max_prob_1 >= threshold_1,
                max_prob_2 >= threshold_2
            )

            filtered_gt_1 = gt_1[filtered_indices]
            filtered_predictions_1 = predictions_1[filtered_indices]
            filtered_gt_2 = gt_2[filtered_indices]
            filtered_predictions_2 = predictions_2[filtered_indices]

            accuracy = (
                           np.logical_and(
                               filtered_gt_1 == filtered_predictions_1,
                               filtered_gt_2 == filtered_predictions_2
                           )
                       ).sum() / len(filtered_gt_1)

            curr_accuracies.append(accuracy)
            curr_dataset_kept.append(len(filtered_gt_1) / len(gt_1))

        accuracies.append(curr_accuracies)
        dataset_kept.append(curr_dataset_kept)

    filename = None
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)
        filename = os.path.join(
            output_directory,
            'confidence_thresholding_2thresholds_3d.' + file_format
        )

    visualization_utils.confidence_fitlering_3d_plot(
        np.array(thresholds),
        np.array(thresholds),
        np.array(accuracies),
        np.array(dataset_kept),
        threshold_fields,
        title='Confidence_Thresholding, two thresholds',
        filename=filename
    )


def binary_threshold_vs_metric(
        probs_per_model,
        gt,
        metrics,
        positive_label=1,
        model_names=None,
        output_directory=None,
        file_format='pdf',
        **kwargs
):
    probs = probs_per_model
    model_names_list = convert_to_list(model_names)
    metrics_list = convert_to_list(metrics)
    filename_template = 'binary_threshold_vs_metric_{}.' + file_format
    filename_template_path = generate_filename_template_path(
        output_directory,
        filename_template
    )

    thresholds = [t / 100 for t in range(0, 101, 5)]

    supported_metrics = {'f1', 'precision', 'recall', 'accuracy'}

    for metric in metrics_list:

        if metric not in supported_metrics:
            logging.error("Metric {} not supported".format(metric))
            continue

        scores = []

        for i, prob in enumerate(probs):

            scores_alg = []

            if len(prob.shape) == 2:
                if prob.shape[1] > positive_label:
                    prob = prob[:, positive_label]
                else:
                    raise Exception(
                        'the specified positive label {} is not '
                        'present in the probabilities'.format(
                            positive_label
                        )
                    )

            for threshold in thresholds:
                threshold = threshold if threshold < 1 else 0.99

                t_gt = gt[prob >= threshold]
                predictions = prob >= threshold
                t_predictions = predictions[prob >= threshold]

                if metric == 'f1':
                    metric_score = sklearn.metrics.f1_score(
                        t_gt,
                        t_predictions
                    )
                elif metric == 'precision':
                    metric_score = sklearn.metrics.precision_score(
                        t_gt,
                        t_predictions
                    )
                elif metric == 'recall':
                    metric_score = sklearn.metrics.recall_score(
                        t_gt,
                        t_predictions
                    )
                elif metric == ACCURACY:
                    metric_score = sklearn.metrics.accuracy_score(
                        t_gt,
                        t_predictions
                    )

                scores_alg.append(metric_score)

            scores.append(scores_alg)

        filename = None
        if output_directory:
            os.makedirs(output_directory, exist_ok=True)
            filename = filename_template_path.format(metric)

        visualization_utils.threshold_vs_metric_plot(
            thresholds,
            scores,
            model_names_list,
            title='Binary threshold vs {}'.format(metric),
            filename=filename
        )


def roc_curves(
        probs_per_model,
        gt,
        positive_label=1,
        model_names=None,
        output_directory=None,
        file_format='pdf',
        **kwargs
):
    probs = probs_per_model
    model_names_list = convert_to_list(model_names)
    fpr_tprs = []

    for i, prob in enumerate(probs):
        if len(prob.shape) > 1:
            prob = prob[:, positive_label]
        fpr, tpr, _ = sklearn.metrics.roc_curve(
            gt, prob,
            pos_label=positive_label
        )
        fpr_tprs.append((fpr, tpr))

    filename = None
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)
        filename = os.path.join(
            output_directory,
            'roc_curves.' + file_format
        )

    visualization_utils.roc_curves(
        fpr_tprs,
        model_names_list,
        title='ROC curves',
        filename=filename
    )


def roc_curves_from_test_statistics(
        test_stats_per_model,
        field,
        model_names=None,
        output_directory=None,
        file_format='pdf',
        **kwargs
):
    model_names_list = convert_to_list(model_names)
    filename_template = 'roc_curves_from_prediction_statistics.' + file_format
    filename_template_path = generate_filename_template_path(
        output_directory,
        filename_template
    )
    fpr_tprs = []
    for curr_test_statistics in test_stats_per_model:
        fpr = curr_test_statistics[field]['roc_curve'][
            'false_positive_rate']
        tpr = curr_test_statistics[field]['roc_curve'][
            'true_positive_rate']
        fpr_tprs.append((fpr, tpr))

    visualization_utils.roc_curves(
        fpr_tprs,
        model_names_list,
        title='ROC curves',
        filename=filename_template_path
    )


def calibration_1_vs_all(
        probabilities,
        ground_truth,
        field,
        top_n_classes,
        labels_limit,
        model_names=None,
        output_directory=None,
        file_format='pdf',
        **kwargs
):
    if len(probabilities) < 1:
        logging.error('No probabilities provided')
        return

    filename_template = None
    if output_directory:
        filename_template = os.path.join(
            output_directory,
            'calibration_1_vs_all_{}.' + file_format
        )

    gt = load_from_file(ground_truth, field)
    if labels_limit > 0:
        gt[gt > labels_limit] = labels_limit
    probs = [load_from_file(probs_fn, dtype=float)
             for probs_fn in probabilities]
    for i, prob in enumerate(probs):
        if labels_limit > 0 and prob.shape[1] > labels_limit + 1:
            prob_limit = prob[:, :labels_limit + 1]
            prob_limit[:, labels_limit] = prob[:, labels_limit:].sum(1)
            probs[i] = prob_limit

    num_classes = max(gt)

    brier_scores = []

    classes = (min(num_classes, top_n_classes[0]) if top_n_classes[0] > 0
               else num_classes)

    for class_idx in range(classes):
        fraction_positives_class = []
        mean_predicted_vals_class = []
        probs_class = []
        brier_scores_class = []
        for prob in probs:
            # ground_truth is an vector of integers, each integer is a class
            # index to have a [0,1] vector we have to check if the value equals
            # the input class index and convert the resulting boolean vector
            # into an integer vector probabilities is a n x c matrix, n is the
            # number of datapoints and c number of classes; its values are the
            # probabilities of the ith datapoint to be classified as belonging
            # to the jth class according to the learned model. For this reason
            # we need to take only the column of predictions that is about the
            # class we are interested in, the input class index

            gt_class = (gt == class_idx).astype(int)
            prob_class = prob[:, class_idx]

            (
                curr_fraction_positives,
                curr_mean_predicted_vals
            ) = calibration_curve(gt_class, prob_class, n_bins=21)

            fraction_positives_class.append(curr_fraction_positives)
            mean_predicted_vals_class.append(curr_mean_predicted_vals)
            probs_class.append(prob[:, class_idx])
            brier_scores_class.append(
                brier_score_loss(
                    gt_class,
                    prob_class, pos_label=1
                )
            )

        brier_scores.append(brier_scores_class)

        filename = None
        if output_directory:
            os.makedirs(output_directory, exist_ok=True)
            filename = filename_template.format(class_idx)

        visualization_utils.calibration_plot(
            fraction_positives_class,
            mean_predicted_vals_class,
            model_names,
            filename=filename
        )

        filename = None
        if output_directory:
            os.makedirs(output_directory, exist_ok=True)
            filename = filename_template.format(
                'prediction_distribution_' + str(class_idx)
            )

        visualization_utils.predictions_distribution_plot(
            probs_class,
            model_names,
            filename=filename
        )

    filename = None
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)
        filename = filename_template.format('brier')

    visualization_utils.brier_plot(
        np.array(brier_scores),
        model_names,
        filename=filename
    )


def calibration_multiclass(
        probabilities,
        ground_truth,
        field,
        labels_limit,
        model_names=None,
        output_directory=None,
        file_format='pdf',
        **kwargs
):
    if len(probabilities) < 1:
        logging.error('No probabilities provided')
        return

    filename_template = None
    if output_directory:
        filename_template = os.path.join(
            output_directory,
            'calibration_multiclass{}.' + file_format
        )

    gt = load_from_file(ground_truth, field)
    if labels_limit > 0:
        gt[gt > labels_limit] = labels_limit

    prob_classes = 0
    probs = [load_from_file(probs_fn, dtype=float)
             for probs_fn in probabilities]

    for i, prob in enumerate(probs):
        if labels_limit > 0 and prob.shape[1] > labels_limit + 1:
            prob_limit = prob[:, :labels_limit + 1]
            prob_limit[:, labels_limit] = prob[:, labels_limit:].sum(1)
            probs[i] = prob_limit
        if probs[i].shape[1] > prob_classes:
            prob_classes = probs[i].shape[1]

    gt_one_hot_dim_2 = max(prob_classes, max(gt) + 1)
    gt_one_hot = np.zeros((len(gt), gt_one_hot_dim_2))
    gt_one_hot[np.arange(len(gt)), gt] = 1
    gt_one_hot_flat = gt_one_hot.flatten()

    fraction_positives = []
    mean_predicted_vals = []
    brier_scores = []
    for prob in probs:
        # flatten probabilities to be compared to flatten ground truth
        prob_flat = prob.flatten()
        curr_fraction_positives, curr_mean_predicted_vals = calibration_curve(
            gt_one_hot_flat,
            prob_flat,
            n_bins=21
        )
        fraction_positives.append(curr_fraction_positives)
        mean_predicted_vals.append(curr_mean_predicted_vals)
        brier_scores.append(
            brier_score_loss(
                gt_one_hot_flat,
                prob_flat,
                pos_label=1
            )
        )

    filename = None
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)
        filename = filename_template.format('')

    visualization_utils.calibration_plot(
        fraction_positives,
        mean_predicted_vals,
        model_names,
        filename=filename
    )

    filename = None
    if output_directory:
        filename = filename_template.format('_brier')

    visualization_utils.compare_classifiers_plot(
        [brier_scores],
        ['brier'],
        model_names,
        adaptive=True,
        decimals=8,
        filename=filename
    )

    for i, brier_score in enumerate(brier_scores):
        if i < len(model_names):
            format_str = '{}: '.format(model_names[i])
            format_str += '{}'
        else:
            format_str = '{}'
        logging.info(format_str.format(brier_score))


def confusion_matrix(
        test_statistics,
        ground_truth_metadata,
        field,
        top_n_classes,
        normalize,
        model_names=None,
        output_directory=None,
        file_format='pdf',
        **kwargs
):
    if len(test_statistics) < 1:
        logging.error('No test_statistics provided')
        return

    filename_template = None
    if output_directory:
        filename_template = os.path.join(
            output_directory,
            'confusion_matrix_{}_{}_{}.' + file_format
        )

    metadata = load_json(ground_truth_metadata)
    test_statistics_per_model_name = [load_json(test_statistics_f)
                                      for test_statistics_f in
                                      test_statistics]

    fields_set = set()
    for ls in test_statistics_per_model_name:
        for key in ls:
            fields_set.add(key)
    fields = [field] if field is not None and len(field) > 0 else fields_set

    for i, test_statistics in enumerate(
            test_statistics_per_model_name):
        for field in fields:
            if 'confusion_matrix' in test_statistics[field]:
                confusion_matrix = np.array(
                    test_statistics[field]['confusion_matrix']
                )
                model_name_name = model_names[i] if (
                        model_names is not None and i < len(model_names)
                ) else ''

                if field in metadata and 'idx2str' in metadata[field]:
                    labels = metadata[field]['idx2str']
                else:
                    labels = list(range(len(confusion_matrix)))

                for k in top_n_classes:
                    k = (min(k, confusion_matrix.shape[0])
                         if k > 0 else confusion_matrix.shape[0])
                    cm = confusion_matrix[:k, :k]
                    if normalize:
                        with np.errstate(divide='ignore', invalid='ignore'):
                            cm_norm = np.true_divide(cm,
                                                     cm.sum(1)[:, np.newaxis])
                            cm_norm[cm_norm == np.inf] = 0
                            cm_norm = np.nan_to_num(cm_norm)
                        cm = cm_norm

                    filename = None
                    if output_directory:
                        os.makedirs(output_directory, exist_ok=True)
                        filename = filename_template.format(
                            model_name_name,
                            field,
                            'top' + str(k)
                        )

                    visualization_utils.confusion_matrix_plot(
                        cm,
                        labels[:k],
                        field=field,
                        filename=filename
                    )

                    entropies = []
                    for row in cm:
                        if np.count_nonzero(row) > 0:
                            entropies.append(entropy(row))
                        else:
                            entropies.append(0)
                    class_entropy = np.array(entropies)
                    class_desc_entropy = np.argsort(class_entropy)[::-1]
                    desc_entropy = class_entropy[class_desc_entropy]

                    filename = None
                    if output_directory:
                        filename = filename_template.format(
                            'entropy_' + model_name_name,
                            field,
                            'top' + str(k)
                        )

                    visualization_utils.bar_plot(
                        class_desc_entropy,
                        desc_entropy,
                        labels=[labels[i] for i in class_desc_entropy],
                        title='Classes ranked by entropy of '
                              'Confusion Matrix row',
                        filename=filename
                    )


def frequency_vs_f1(
        test_statistics,
        ground_truth_metadata,
        field,
        top_n_classes,
        model_names=None,
        output_directory=None,
        file_format='pdf',
        **kwargs
):
    if len(test_statistics) < 1:
        logging.error('No test_statistics provided')
        return

    filename_template = None
    if output_directory:
        filename_template = os.path.join(
            output_directory,
            'frequency_vs_f1_{}_{}.' + file_format
        )

    metadata = load_json(ground_truth_metadata)
    test_statistics_per_model_name = [load_json(test_statistics_f)
                                      for test_statistics_f in
                                      test_statistics]
    k = top_n_classes[0]

    fields_set = set()
    for ls in test_statistics_per_model_name:
        for key in ls:
            fields_set.add(key)
    fields = [field] if field is not None and len(field) > 0 else fields_set

    for i, test_statistics in enumerate(
            test_statistics_per_model_name):
        for field in fields:
            model_name_name = (model_names[i]
                               if model_names is not None and i < len(
                model_names)
                               else '')
            per_class_stats = test_statistics[field]['per_class_stats']
            f1_scores = []
            labels = []
            class_names = metadata[field]['idx2str']
            if k > 0:
                class_names = class_names[:k]
            for class_name in class_names:
                class_stats = per_class_stats[class_name]
                f1_scores.append(class_stats['f1_score'])
                labels.append(class_name)

            f1_np = np.nan_to_num(np.array(f1_scores, dtype=np.float32))
            f1_sorted_indices = f1_np.argsort()

            field_frequency_dict = {
                metadata[field]['str2idx'][key]: val
                for key, val in metadata[field]['str2freq'].items()
            }
            field_frequency_np = np.array(
                [field_frequency_dict[class_id]
                 for class_id in sorted(field_frequency_dict)],
                dtype=np.int32
            )

            field_frequency_reordered = field_frequency_np[
                                            f1_sorted_indices[::-1]
                                        ][:len(f1_sorted_indices)]
            f1_reordered = f1_np[f1_sorted_indices[::-1]][
                           :len(f1_sorted_indices)]

            filename = None
            if output_directory:
                os.makedirs(output_directory, exist_ok=True)
                filename = filename_template.format(model_name_name, field)

            visualization_utils.double_axis_line_plot(
                f1_reordered,
                field_frequency_reordered,
                'F1 score',
                'frequency',
                labels=labels,
                title='{} F1 Score vs Frequency {}'.format(
                    model_name_name,
                    field
                ),
                filename=filename
            )

            frequency_sorted_indices = field_frequency_np.argsort()
            field_frequency_reordered = field_frequency_np[
                                            frequency_sorted_indices[::-1]
                                        ][:len(f1_sorted_indices)]

            f1_reordered = np.zeros(len(field_frequency_reordered))

            for idx in frequency_sorted_indices[::-1]:
                if idx < len(f1_np):
                    f1_reordered[idx] = f1_np[idx]

            visualization_utils.double_axis_line_plot(
                field_frequency_reordered,
                f1_reordered,
                'frequency',
                'F1 score',
                labels=labels,
                title='{} F1 Score vs Frequency {}'.format(
                    model_name_name,
                    field
                ),
                filename=filename
            )


def cli(sys_argv):
    parser = argparse.ArgumentParser(
        description='This script analyzes results and shows some nice plots.',
        prog='ludwig visualize',
        usage='%(prog)s [options]')

    parser.add_argument('-d', '--data_csv', help='raw data file')
    parser.add_argument('-g', '--ground_truth', help='ground truth file')
    parser.add_argument(
        '-gm',
        '--ground_truth_metadata',
        help='input metadata JSON file'
    )

    parser.add_argument(
        '-od',
        '--output_directory',
        help='directory where to save plots.'
             'If not specified, plots will be displayed in a window'
    )
    parser.add_argument(
        '-ff',
        '--file_format',
        help='file format of output plots',
        default='pdf',
        choices=['pdf', 'png']
    )

    parser.add_argument(
        '-v',
        '--visualization',
        default='confidence_thresholding',
        choices=['learning_curves',
                 'compare_performance',
                 'compare_classifiers_performance_from_prob',
                 'compare_classifiers_performance_from_pred',
                 'compare_classifiers_performance_subset',
                 'compare_classifiers_performance_changing_k',
                 'compare_classifiers_multiclass_multimetric',
                 'compare_classifiers_predictions',
                 'compare_classifiers_predictions_distribution',
                 'confidence_thresholding',
                 'confidence_thresholding_data_vs_acc',
                 'confidence_thresholding_data_vs_acc_subset',
                 'confidence_thresholding_data_vs_acc_subset_per_class',
                 'confidence_thresholding_2thresholds_2d',
                 'confidence_thresholding_2thresholds_3d',
                 'binary_threshold_vs_metric',
                 'roc_curves',
                 'roc_curves_from_test_statistics',
                 'calibration_1_vs_all',
                 'calibration_multiclass',
                 'confusion_matrix',
                 'frequency_vs_f1'],
        help='type of visualization'
    )

    parser.add_argument(
        '-f',
        '--field',
        default=[],
        help='field containing ground truth'
    )
    parser.add_argument(
        '-tf',
        '--threshold_fields',
        default=[],
        nargs='+',
        help='fields for 2d threshold'
    )
    parser.add_argument(
        '-pred',
        '--predictions',
        default=[],
        nargs='+',
        type=str,
        help='predictions files'
    )
    parser.add_argument(
        '-prob',
        '--probabilities',
        default=[],
        nargs='+',
        type=str,
        help='probabilities files'
    )
    parser.add_argument(
        '-trs',
        '--training_statistics',
        default=[],
        nargs='+',
        type=str,
        help='training stats files'
    )
    parser.add_argument(
        '-tes',
        '--test_statistics',
        default=[],
        nargs='+',
        type=str,
        help='test stats files'
    )
    parser.add_argument(
        '-mn',
        '--model_names',
        default=[],
        nargs='+',
        type=str,
        help='names of the models to use as labels'
    )
    parser.add_argument(
        '-tn',
        '--top_n_classes',
        default=[0],
        nargs='+',
        type=int,
        help='number of classes to plot'
    )
    parser.add_argument(
        '-k',
        '--top_k',
        default=3,
        type=int,
        help='number of elements in the ranklist to consider'
    )
    parser.add_argument(
        '-ll',
        '--labels_limit',
        default=0,
        type=int,
        help='maximum numbers of labels. '
             'If labels in dataset are higher than this number, "rare" label'
    )
    parser.add_argument(
        '-ss',
        '--subset',
        default='ground_truth',
        choices=['ground_truth', PREDICTIONS],
        help='type of subset filtering'
    )
    parser.add_argument(
        '-n',
        '--normalize',
        action='store_true',
        default=False,
        help='normalize rows in confusion matrix'
    )
    parser.add_argument(
        '-m',
        '--metrics',
        default=['f1'],
        nargs='+',
        type=str,
        help='metrics to dispay in threshold_vs_metric'
    )
    parser.add_argument(
        '-pl',
        '--positive_label',
        type=int,
        default=1,
        help='label of the positive class for the roc curve'
    )
    parser.add_argument(
        '-l',
        '--logging_level',
        default='info',
        help='the level of logging to use',
        choices=['critical', 'error', 'warning', 'info', 'debug', 'notset']
    )

    args = parser.parse_args(sys_argv)

    logging.basicConfig(
        stream=sys.stdout,
        level=logging_level_registry[args.logging_level],
        format='%(message)s'
    )

    if args.visualization == 'compare_performance':
        test_stats_per_model = load_data_for_viz(
            'load_json', vars(args)['test_statistics']
        )
        compare_performance(
            test_stats_per_model, **vars(args)
        )
    elif args.visualization == 'compare_classifiers_performance_from_prob':
        gt = load_from_file(vars(args)['ground_truth'], vars(args)['field'])
        probabilities_per_model = load_data_for_viz(
            'load_from_file', vars(args)['probabilities'], dtype=float
        )
        compare_classifiers_performance_from_prob(
            probabilities_per_model, gt, **vars(args)
        )
    elif args.visualization == 'compare_classifiers_performance_from_pred':
        gt = load_from_file(vars(args)['ground_truth'], vars(args)['field'])
        metadata = load_json(vars(args)['ground_truth_metadata'])
        preds_per_model = load_data_for_viz(
            'load_from_file', vars(args)['predictions'], dtype=str
        )
        compare_classifiers_performance_from_pred(
            preds_per_model, gt, metadata, **vars(args)
        )
    elif args.visualization == 'compare_classifiers_performance_subset':
        gt = load_from_file(vars(args)['ground_truth'], vars(args)['field'])
        probabilities_per_model = load_data_for_viz(
            'load_from_file', vars(args)['probabilities'], dtype=float
        )
        compare_classifiers_performance_subset(
            probabilities_per_model, gt, **vars(args)
        )
    elif args.visualization == 'compare_classifiers_performance_changing_k':
        gt = load_from_file(vars(args)['ground_truth'], vars(args)['field'])
        probabilities_per_model = load_data_for_viz(
            'load_from_file', vars(args)['probabilities'], dtype=float
        )
        compare_classifiers_performance_changing_k(
            probabilities_per_model, gt, **vars(args)
        )
    elif args.visualization == 'compare_classifiers_multiclass_multimetric':
        test_stats_per_model = load_data_for_viz(
            'load_json', vars(args)['test_statistics']
        )
        metadata = load_json(vars(args)['ground_truth_metadata'])
        compare_classifiers_multiclass_multimetric(
            test_stats_per_model=test_stats_per_model, metadata=metadata, **vars(args)
        )
    elif args.visualization == 'compare_classifiers_predictions':
        gt = load_from_file(vars(args)['ground_truth'], vars(args)['field'])
        preds_per_model = load_data_for_viz(
            'load_from_file', vars(args)['predictions'], dtype=str
        )
        compare_classifiers_predictions(
            preds_per_model, gt, **vars(args)
        )
    elif args.visualization == 'compare_classifiers_predictions_distribution':
        gt = load_from_file(vars(args)['ground_truth'], vars(args)['field'])
        preds_per_model = load_data_for_viz(
            'load_from_file', vars(args)['predictions'], dtype=str
        )
        compare_classifiers_predictions_distribution(
            preds_per_model, gt, **vars(args)
        )
    elif args.visualization == 'confidence_thresholding':
        gt = load_from_file(vars(args)['ground_truth'], vars(args)['field'])
        probabilities_per_model = load_data_for_viz(
            'load_from_file', vars(args)['probabilities'], dtype=float
        )
        confidence_thresholding(
            probabilities_per_model, gt, **vars(args)
        )
    elif args.visualization == 'confidence_thresholding_data_vs_acc':
        gt = load_from_file(vars(args)['ground_truth'], vars(args)['field'])
        probabilities_per_model = load_data_for_viz(
            'load_from_file', vars(args)['probabilities'], dtype=float
        )
        confidence_thresholding_data_vs_acc(
            probabilities_per_model, gt, **vars(args)
        )
    elif args.visualization == 'confidence_thresholding_data_vs_acc_subset':
        gt = load_from_file(vars(args)['ground_truth'], vars(args)['field'])
        probabilities_per_model = load_data_for_viz(
            'load_from_file', vars(args)['probabilities'], dtype=float
        )
        confidence_thresholding_data_vs_acc_subset(
            probabilities_per_model, gt, **vars(args)
        )
    elif (args.visualization ==
          'confidence_thresholding_data_vs_acc_subset_per_class'):
        gt = load_from_file(vars(args)['ground_truth'], vars(args)['field'])
        metadata = load_json(vars(args)['ground_truth_metadata'])
        probabilities_per_model = load_data_for_viz(
            'load_from_file', vars(args)['probabilities'], dtype=float
        )
        confidence_thresholding_data_vs_acc_subset_per_class(
            probabilities_per_model, gt, metadata, **vars(args)
        )
    elif args.visualization == 'confidence_thresholding_2thresholds_2d':
        gt1 = load_from_file(
            vars(args)['ground_truth'],
            vars(args)['threshold_fields'][0]
        )
        gt2 = load_from_file(
            vars(args)['ground_truth'],
            vars(args)['threshold_fields'][1]
        )
        probabilities_per_model = load_data_for_viz(
            'load_from_file', vars(args)['probabilities'], dtype=float
        )
        confidence_thresholding_2thresholds_2d(
            probabilities_per_model, [gt1, gt2], **vars(args)
        )
    elif args.visualization == 'confidence_thresholding_2thresholds_3d':
        gt1 = load_from_file(
            vars(args)['ground_truth'],
            vars(args)['threshold_fields'][0]
        )
        gt2 = load_from_file(
            vars(args)['ground_truth'],
            vars(args)['threshold_fields'][1]
        )
        probabilities_per_model = load_data_for_viz(
            'load_from_file', vars(args)['probabilities'], dtype=float
        )
        confidence_thresholding_2thresholds_3d(
            probabilities_per_model, [gt1, gt2], **vars(args)
        )
    elif args.visualization == 'binary_threshold_vs_metric':
        gt = load_from_file(vars(args)['ground_truth'], vars(args)['field'])
        probabilities_per_model = load_data_for_viz(
            'load_from_file', vars(args)['probabilities'], dtype=float
        )
        binary_threshold_vs_metric(
            probabilities_per_model, gt, **vars(args)
        )
    elif args.visualization == 'roc_curves':
        gt = load_from_file(vars(args)['ground_truth'], vars(args)['field'])
        probabilities_per_model = load_data_for_viz(
            'load_from_file', vars(args)['probabilities'], dtype=float
        )
        roc_curves(
            probabilities_per_model, gt, **vars(args)
        )
    elif args.visualization == 'roc_curves_from_test_statistics':
        test_stats_per_model = load_data_for_viz(
            'load_json', vars(args)['test_statistics']
        )
        roc_curves_from_test_statistics(
            test_stats_per_model, **vars(args)
        )
    elif args.visualization == 'calibration_1_vs_all':
        calibration_1_vs_all(**vars(args))
    elif args.visualization == 'calibration_multiclass':
        calibration_multiclass(**vars(args))
    elif args.visualization == 'confusion_matrix':
        confusion_matrix(**vars(args))
    elif args.visualization == 'frequency_vs_f1':
        frequency_vs_f1(**vars(args))
    elif args.visualization == 'learning_curves':
        train_stats_per_model = load_data_for_viz(
            'load_json', vars(args)['training_statistics']
        )
        learning_curves(
            train_stats_per_model, **vars(args)
        )
    else:
        logging.info('Visualization argument not recognized')


if __name__ == '__main__':
    contrib_command("visualize", *sys.argv)
    cli(sys.argv[1:])
