import abc
import multiprocessing

import tensorflow as tf
import pandas as pd


class Dataset(metaclass=abc.ABCMeta):
    """Abstract base class as the interface that run_experiment expects."""
    @abc.abstractmethod
    def create_input_fn(self):
        pass


class PandasDataset(Dataset):
    """Wraps a pd.DataFrame to provide a uniform create_input_fn method."""
    def __init__(self, df, label_column_name):
        """
        Initializer.
        :param df: pd.DataFrame with label column.
        :param label_column_name: label column key.
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("'df' must be a pd.DataFrame.")
        self._df = df
        self._label_column_name = label_column_name

    def create_input_fn(self, batch_size=128, shuffle=True, num_epochs=None):
        x = self._df.drop([self._label_column_name], axis="columns")
        y = self._df[self._label_column_name]
        return tf.estimator.inputs.pandas_input_fn(
            x=x,
            y=y,
            shuffle=shuffle,
            batch_size=batch_size,
            num_epochs=num_epochs)


class NumpyDataset(Dataset):
    """Wraps a numpy array or dictionary of numpy arrays to provide a uniform create_input_fn method."""
    def __init__(self, inputs, labels):
        """
        Initializer.
        :param inputs: np.array or dict: name -> np.array.
        :param labels: np.array.
        """
        if not isinstance(inputs, dict):
            inputs = {"X": inputs}
        self._inputs = inputs
        self._labels = labels

    def create_input_fn(self, batch_size=128, shuffle=True, num_epochs=None):
        return tf.estimator.inputs.numpy_input_fn(
            x=self._inputs,
            y=self._labels,
            shuffle=shuffle,
            batch_size=batch_size,
            num_epochs=num_epochs)


class CsvDataset(Dataset):
    """Wraps one or more csv files to provide a uniform create_input_fn method."""
    def __init__(self, files, column_names, label_column, unused_columns=[]):
        if isinstance(files, str):
            files = [files]
        self._csv_files = files
        self._column_names = column_names  # Make sure its an (ordered) list of strings
        self._label_column = label_column
        self._unused_columns = unused_columns
        # TODO: rename feature columns to column names here

    def create_input_fn(self, batch_size=128, shuffle=True, num_epochs=None):
        csv_files = self._csv_files
        feature_columns = self._column_names
        label_column = self._label_column
        unused_columns = self._unused_columns
        column_default_values = [[""] for _ in feature_columns]  # Missing values not allowed.

        def parse_csv(rows):
            column_values = tf.decode_csv(rows, record_defaults=column_default_values)
            data = {name: val for name, val in zip(feature_columns, column_values)}
            for col in unused_columns:
                data.pop(col)
            #for key, value in data.items():
            #    data[key] = tf.expand_dims(data[key], -1)
            return data

        def input_fn():
            dataset = tf.data.TextLineDataset(csv_files).map(parse_csv)

            if shuffle:
                dataset = dataset.shuffle(buffer_size=10 * batch_size)
            data = (dataset
                    .repeat(num_epochs)
                    .batch(batch_size)
                    .make_one_shot_iterator().get_next())
            labels = data.pop(label_column)
            features = data
            return features, labels

        return input_fn


# TODO: dataset.prefetch()
# TODO: Use tf.Dataset for tfrecords too?


class TfrecordsDataset(Dataset):
    """Wraps one or more tfrecords files to provide a uniform create_input_fn method."""
    def __init__(self, files, feature_columns, label_column_name):
        if isinstance(files, str):
            files = [files]
        self._record_files = files
        self._feature_columns = feature_columns
        self._label_column_name = label_column_name

    def create_input_fn(self, batch_size=128, shuffle=True, num_epochs=None):
        record_files = self._record_files
        feature_columns = self._feature_columns
        label_column_name = self._label_column_name

        capacity = 20 * batch_size  # TODO: Tune some values for queues etc for performance.

        def input_fn():
            filename_queue = tf.train.string_input_producer(
                record_files,
                num_epochs=num_epochs,
                shuffle=shuffle)
            record_reader = tf.TFRecordReader()
            _, serialized_examples = record_reader.read_up_to(filename_queue, batch_size)

            if shuffle:
                serialized_batch = tf.train.shuffle_batch(
                    [serialized_examples],
                    batch_size=batch_size,
                    capacity=capacity,
                    min_after_dequeue=10 * batch_size,
                    num_threads=multiprocessing.cpu_count(),
                    enqueue_many=True,
                    allow_smaller_final_batch=True)
            else:
                serialized_batch = tf.train.batch(
                    [serialized_examples],
                    batch_size=batch_size,
                    capacity=capacity,
                    num_threads=multiprocessing.cpu_count(),
                    enqueue_many=True,
                    allow_smaller_final_batch=True)

            examples = tf.parse_example(
                serialized_batch,
                tf.feature_column.make_parse_example_spec(feature_columns=feature_columns))

            # TODO: preprocessing options.

            if label_column_name is not None:
                labels = examples.pop(label_column_name)
                features = examples
                return features, labels
            else:
                return examples

        return input_fn


#def create_dataset(datasource, feature_columns, label_column_name):
#    if isinstance(datasource, pd.DataFrame):
#        return _PandasDataset(datasource, label_column_name)
#    else:
#        raise ValueError("Unsupported datasource.")
