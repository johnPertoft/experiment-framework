import os
import re
import json
import math
import datetime
import warnings

import tensorflow as tf


class ExperimentResult:
    def __init__(self, hparams, config, train_results, eval_results, test_results):
        self._hparams = hparams
        self._config = {"model_dir": config.model_dir, "tf_seed": config.tf_random_seed}
        self._train_results = train_results
        self._eval_results = eval_results
        self._test_results = test_results

    @property
    def results(self):
        return self._test_results

    @property
    def train_results(self):
        return self._train_results

    @property
    def eval_results(self):
        return self._eval_results

    @property
    def config(self):
        return self._config

    @property
    def hparams(self):
        return self._hparams

    def serialize(self, path):
        """
        Serialize hparams, config, and results as json.
        :param path: file path.
        """
        contents = {
            "hparams": self.hparams,
            "config": self.config,
            "results": {k: float(v) for k, v in self.results.items()}
        }
        with open(path, "w") as f:
            json.dump(contents, f, sort_keys=True, indent=2)


def _default_config():
    return tf.estimator.RunConfig(
        tf_random_seed=123)


def _default_model_dir(description):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
    return "{}_{}".format(description, timestamp)


_CHECKPOINT_EXTRACTOR_REGEX = re.compile("all_model_checkpoint_paths: \"(.*)\"")
def _checkpoint_paths(model_dir):
    def get_path(l):
        local_path = _CHECKPOINT_EXTRACTOR_REGEX.search(l).group(1)
        return os.path.join(model_dir, local_path)

    with open(os.path.join(model_dir, "checkpoint")) as checkpoints_file:
        next(checkpoints_file)  # Skip first line (default checkpoint).
        return [get_path(line) for line in checkpoints_file]


def run_experiment(estimator_fn,
                   hparams,
                   train_data,
                   validation_data,
                   test_data,
                   max_steps,
                   model_dir=None,
                   description=None,
                   config=None,
                   train_steps_per_evaluation=500,
                   early_stopping_fn=None,
                   steps_before_early_stopping=0):

    if model_dir is None:
        if description is None:
            raise ValueError("Must pass 'description' argument if 'model_dir' is not given.")
        model_dir = _default_model_dir(description)
    else:
        if description is not None:
            warnings.warn("'model_dir' was passed so 'description' will not be used.", RuntimeWarning)

    config = config or _default_config()

    estimator = estimator_fn(model_dir, hparams, config)

    # TODO: Fix the extra checkpoint on start and end of each train(), e.g. 1001 and 2000.

    tf.logging.set_verbosity(tf.logging.INFO)  # TODO: Need this?

    train_input_fn = train_data.create_input_fn(hparams["batch_size"])
    val_input_fn = validation_data.create_input_fn(shuffle=False, num_epochs=1)

    if steps_before_early_stopping > 0:
        estimator.train(train_input_fn, steps=steps_before_early_stopping)

    should_early_stop = early_stopping_fn if early_stopping_fn is not None else lambda *_: False

    # Run training/evaluation loop for fixed number of steps.
    steps = max(0, max_steps - steps_before_early_stopping)
    train_eval_iterations = int(math.ceil(steps / train_steps_per_evaluation))
    prev_eval_metrics = []
    best_checkpoint_path = tf.train.latest_checkpoint(estimator.model_dir)
    for _ in range(train_eval_iterations):
        estimator.train(train_input_fn, steps=train_steps_per_evaluation)
        best_checkpoint_path = tf.train.latest_checkpoint(estimator.model_dir)
        eval_metrics = estimator.evaluate(val_input_fn)

        if should_early_stop(eval_metrics, prev_eval_metrics):
            # Revert checkpoint path to previous one.
            best_checkpoint_path = _checkpoint_paths(estimator.model_dir)[-2]
            eval_metrics = prev_eval_metrics[-1]
            break
        prev_eval_metrics.append(eval_metrics)

    # Set the best checkpoint as default for further model use.
    tf.train.update_checkpoint_state(estimator.model_dir, best_checkpoint_path)

    train_metrics = estimator.evaluate(
        train_data.create_input_fn(shuffle=False, num_epochs=1),
        checkpoint_path=best_checkpoint_path,
        name="train_data")

    test_metrics = estimator.evaluate(
        test_data.create_input_fn(shuffle=False, num_epochs=1),
        checkpoint_path=best_checkpoint_path,
        name="test_data")

    experiment_result = ExperimentResult(
        hparams=hparams,
        config=estimator.config,
        train_results=train_metrics,
        eval_results=eval_metrics,
        test_results=test_metrics)

    experiment_result.serialize(os.path.join(estimator.model_dir, "experiment_results"))

    return experiment_result, estimator


def tune_experiment(estimator_fn,
                    hparams_tuner,
                    train_data,
                    validation_data,
                    test_data,
                    model_dir=None,
                    description=None,
                    config=None):
    # TODO: Should take a tuner object which can implement different hyper param search strategies.
    pass
