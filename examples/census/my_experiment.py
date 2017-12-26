import operator
import functools

import experiments
import tensorflow as tf

from features import RAW_FEATURE_COLUMNS, LABEL_VALUES


(age, education_num, capital_gain, capital_loss, hours_per_week,
 gender, race, education, marital_status, relationship, workclass,
 occupation, native_country) = RAW_FEATURE_COLUMNS

feature_columns = [
    # Doing nothing for numeric columns.
    age,
    education_num,
    capital_gain,
    capital_loss,
    hours_per_week,

    # One hot encoded categorical columns with few classes.
    tf.feature_column.indicator_column(gender),
    tf.feature_column.indicator_column(race),
    tf.feature_column.indicator_column(education),
    tf.feature_column.indicator_column(marital_status),
    tf.feature_column.indicator_column(relationship),
    tf.feature_column.indicator_column(workclass),

    # Embeddings for categorical columns with many classes.
    tf.feature_column.embedding_column(occupation, dimension=8),
    tf.feature_column.embedding_column(native_country, dimension=8)
]

# Ordered as given by csv file's columns.
column_names = ["age", "workclass", "fnlwgt", "education", "education_num",
               "marital_status", "occupation", "relationship", "race", "gender",
               "capital_gain", "capital_loss", "hours_per_week",
               "native_country", "income_bracket"]
label_column = "income_bracket"
unused_columns = ["fnlwgt"]


csv_dataset = functools.partial(
    experiments.CsvDataset,
    feature_columns=column_names,
    label_column=label_column,
    unused_columns=unused_columns)
train_data = csv_dataset("train.csv")
val_data = csv_dataset("val.csv")
test_data = csv_dataset("test.csv")


@experiments.model
def my_dnn(mode, features, labels, hparams, config):
    inputs = tf.feature_column.input_layer(features, feature_columns)
    table = tf.contrib.lookup.index_table_from_tensor(tf.constant(LABEL_VALUES))  # Why cant we use a feature column for label as well?
    labels = table.lookup(labels)

    net = inputs
    net = tf.layers.dense(net, 2048, activation=tf.nn.relu)
    net = tf.layers.dense(net, 1024, activation=tf.nn.relu)
    logits = tf.layers.dense(net, 1, activation=None)
    logits = tf.squeeze(logits, axis=-1)

    probabilities = None
    predictions = None
    if mode in (tf.estimator.ModeKeys.PREDICT, tf.estimator.ModeKeys.EVAL):
        probabilities = tf.nn.sigmoid(logits)
        predictions = tf.cast(probabilities > 0.5, tf.int64)
        # TODO: reverse mapping here possibly?

    loss = None
    if mode in (tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL):
        loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf.cast(labels, tf.float32)))

    train_op = None
    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_or_create_global_step()
        train_op = tf.train.AdamOptimizer(1e-4).minimize(loss, global_step=global_step)

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
    "batch_size": 128
}

result, _ = experiments.run_experiment(
    estimator_fn=my_dnn,
    hparams=hparams,
    train_data=train_data,
    validation_data=val_data,
    test_data=test_data,
    model_dir="runs/test",
    max_steps=15000,
    #steps_before_early_stopping=8000,
    early_stopping_fn=experiments.early_stopping("accuracy", operator.le))

print(result.results)
