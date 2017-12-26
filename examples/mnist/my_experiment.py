import functools

import experiments
from experiments.dataset import TfrecordsDataset
import tensorflow as tf


feature_columns = [
    tf.feature_column.numeric_column("image", dtype=tf.float32, shape=[28, 28]),
    tf.feature_column.numeric_column("label", dtype=tf.int64, shape=[])
]


train_data = TfrecordsDataset("data/mnist.train", feature_columns, "label")
val_data = TfrecordsDataset("data/mnist.eval", feature_columns, "label")
test_data = TfrecordsDataset("data/mnist.test", feature_columns, "label")


@experiments.model
def my_cnn(mode, features, labels, hparams, config):
    X = features["image"]
    X = X[..., tf.newaxis]  # NHWC format.

    N_CLASSES = 10

    # Define building blocks.
    conv2d = functools.partial(tf.layers.conv2d, kernel_size=(3, 3), padding="same", activation=tf.nn.relu)
    maxpool2d = functools.partial(tf.layers.max_pooling2d, pool_size=(2, 2), strides=(2, 2), padding="valid")

    # VGG-16 like architecture.
    net = X

    net = conv2d(net, filters=64)
    net = conv2d(net, filters=64)
    net = maxpool2d(net)

    net = conv2d(net, filters=128)
    net = conv2d(net, filters=128)
    net = maxpool2d(net)

    net = conv2d(net, filters=256)
    net = conv2d(net, filters=256)
    net = maxpool2d(net)

    net = conv2d(net, filters=512)
    net = conv2d(net, filters=512)
    net = maxpool2d(net)

    net = tf.layers.dense(net, units=1024, activation=tf.nn.relu)
    net = tf.layers.dense(net, units=1024, activation=tf.nn.relu)
    logits = tf.layers.dense(net, units=N_CLASSES, activation=None)
    logits = tf.squeeze(logits, axis=[1, 2])

    probabilities = None
    predictions = None
    if mode in (tf.estimator.ModeKeys.PREDICT, tf.estimator.ModeKeys.EVAL):
        probabilities = tf.nn.softmax(logits)
        predictions = tf.argmax(probabilities, axis=1)

    loss = None
    if mode in (tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL):
        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))

    train_op = None
    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_or_create_global_step()
        lr = hparams["learning_rate"]
        train_op = tf.train.AdamOptimizer(lr).minimize(loss, global_step=global_step)

    eval_metric_ops = None
    if mode == tf.estimator.ModeKeys.EVAL:
        eval_metric_ops = {
            "accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions)
        }

    return tf.estimator.EstimatorSpec(
        mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops)


hparams = {
    "batch_size": 128,
    "activation": "relu",
    "dropout": 0.5,
    "optimizer": "Adam",
    "learning_rate": 1e-4
}

result, _ = experiments.run_experiment(
    estimator_fn=my_cnn,
    hparams=hparams,
    train_data=train_data,
    validation_data=val_data,
    test_data=test_data,
    model_dir="runs/test",
    max_steps=5000)

print(result.results)
