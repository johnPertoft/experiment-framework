import tensorflow as tf

import experiments


@experiments.model
def dnn(mode, features, labels, hparams, config):
    predictions = None
    loss = None
    train_op = None
    eval_metric_ops = None

    return tf.estimator.EstimatorSpec(
        mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops)
