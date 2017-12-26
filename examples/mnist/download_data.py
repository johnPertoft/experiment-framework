import os
import shutil
import uuid

import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets import mnist
import numpy as np


def write_tfrecords(tfrecords_path, dataset):
    with tf.python_io.TFRecordWriter(tfrecords_path) as record_writer:
        for image, label in zip(dataset.images, dataset.labels):
            example = tf.train.Example(features=tf.train.Features(feature={
                #"image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image.tostring()])),
                "image": tf.train.Feature(float_list=tf.train.FloatList(value=image.flatten().astype(np.float32))),
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[int(label)]))
            }))
            record_writer.write(example.SerializeToString())


dataset_dir = "data/"
compressed_files_dir = str(uuid.uuid4())

datasets = mnist.read_data_sets(compressed_files_dir, dtype=tf.uint8, reshape=True, validation_size=10000)

write_tfrecords(os.path.join(dataset_dir, "mnist.train"), datasets.train)
write_tfrecords(os.path.join(dataset_dir, "mnist.eval"), datasets.validation)
write_tfrecords(os.path.join(dataset_dir, "mnist.test"), datasets.test)

shutil.rmtree(compressed_files_dir)