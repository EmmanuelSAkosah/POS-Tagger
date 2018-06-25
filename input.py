from __future__ import print_function
from __future__ import absolute_import
from __future__ import division



import tensorflow as tf
from tensorflow.contrib import training

#Constants
VOCAB_SIZE = 4273
TAGS_SIZE = 148
BUCKET_BOUNDARIES = [11, 21, 31, 41, 51]


def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _,serialized = reader.read(filename_queue)
    # featyre is a dict , which maps features to tensor values
    features = tf.parse_single_example(serialized,
                                      features={
                                          "context": tf.FixedLenFeature([],dtype=tf.string),
                                          "tag":tf.FixedLenFeatures([],dtype=tf.string),
                                          "sequence_length": tf.FixedLenFeature([],dtype=tf.int64)
                                          })
    sequence_length = tf.cast(features['sequence_length'],tf.int32)
    context = tf.decode_raw(features['context'],tf.int64)
    tag = tf.decode_raw(features['tag'],tf.int64)
    return context,tag,sequence_length


def input_producer(context, tag, sequence_length, batch_size, capacity):
    sequence_lengths, (contexts, tags) = training.bucket_by_sequence_length(input_length=sequence_length,
                                                                           tensors=[context,tag],
                                                                           batch_size=batch_size,
                                                                           bucket_boundaries=BUCKET_BOUNDARIES,
                                                                           dynamic_pad=True,
                                                                           capacity=capacity)
    return contexts,tags,sequence_lengths


class Inputs(object):
    def __init__(self,train=True):
     if train is True:
         filename_queue = tf.train.string_input_producer(["records/train.tfrecords"])  # creates strings from tensor of strings
         context, tag, sequence_length = read_and_decode(filename_queue)
         self.context, self.tags, self.sequence_lengths = input_producer(context,
                                                                         tag,
                                                                         sequence_length,
                                                                         batch_size=96,
                                                                         capacity=400)
     #else: #same??






     self.vocab_size = VOCAB_SIZE
     self.tags_size = TAGS_SIZE



class Config(object):
    def __init__(self,train=True):
        if train is True:
            self.embedding_size = 128
            self.num_units = 32
            self.learning_rate = 5e-4
            self.forward_units = 24
            self.backward_units =32
        else:
            self.embedding_size = 24
            self.num_units = 26
            self.learning_rate = 5e-2
            self.forward_units = 2
            self.backward_units =2






















