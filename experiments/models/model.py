import functools

import tensorflow as tf


def model(user_model_fn):
    """
    Decorator to turn the given user_model_fn into a function with a signature that
    run_experiment expects.
    :param user_model_fn: a function with the following signature
        :param mode: tf.estimator.ModeKeys.
        :param features: input features.
        :param labels: labels.
        :param params: dict.
        :param config: tf.estimator.RunConfig.
        :return: tf.estimator.EstimatorSpec.
    :return: a function with the following signature
        :param model_dir: model_dir.
        :param hparams: params that will be passed to the user_model_fn.
        :param config: config that will be passed to the user_model_fn.
        
    Example usage:
    @experiments.model
    def my_dnn(mode, features, labels, hparams, config):
        # Model definition here.
        return tf.estimator.EstimatorSpec(...)
    """
    @functools.wraps(user_model_fn)
    def estimator_fn(model_dir, hparams, config):
        def model_fn(mode, features, labels, params, config):
            return user_model_fn(mode, features, labels, params, config)
        return tf.estimator.Estimator(model_fn, model_dir, config, hparams)

    return estimator_fn
