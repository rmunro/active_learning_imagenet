# Copyright 2017 Robert Munro, with additional code from sources as indicated below.
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

"""Active Learning for ImageNet 

This code is designed as a coding exercise, for use in a job interview or similar situation.

The code runs, but IS DELIBERATELY A BAD IMPLEMENTATION OF ACTIVE LEARNING

The exercise is to improve the code so that it:
 1) takes collection of unlabled images
 2) attempts to classify each image with Inception trained on ImageNet (2012) 
 3) orders the images according to how ecch should be manually labled, to create the best possible training data from the new images.

Steps 1 and 2 are implemented (but could be improved).

Step 3 currently orders the images from the most-to-least confidently classified, which is a bad strategy.

There are many extensions to this code, from a 1 hour exercise to improve how confidence is used, to a multiple week exercice that included retraining all/parts of the model and provided interfaces that were optimal for retraining.

The code is in the style of TensorFlow tutorials and is adapted with thanks to the original authors of:
  https://github.com/tensorflow/models/blob/master/tutorials/image/imagenet/classify_image.py

The code can be run in the same tutorial folder (although not required):
  https://github.com/tensorflow/models/tree/master/tutorials/image/imagenet

To install TensorFlow and for more context on this problem, see: 
  https://www.tensorflow.org/tutorials/image_recognition
In short, you can clone tensorflow at:
  git clone https://github.com/tensorflow/models
And then find the location of the tutorial at:
  cd models/tutorials/image/imagenet
If you are on a Mac, you might need to install tensorflow with the following command:
  sudo -H pip install tensorflow --upgrade --ignore-installed

Usage:
  python active_learning_for_images.py --directory=DIRECTORY_OF_IMAGES

Where DIRECTORY_OF_IMAGES is the directory containing the JPGs you want apply Active Learning to.

Output:
  a printed listed of image names, ordered by how important they are for the training data. 

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import re
import sys
import tarfile

import numpy as np
from six.moves import urllib
import tensorflow as tf

FLAGS = None

# pylint: disable=line-too-long
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
# pylint: enable=line-too-long


class NodeLookup(object):
  """Converts integer node ID's to human readable labels."""

  def __init__(self,
               label_lookup_path=None,
               uid_lookup_path=None):
    if not label_lookup_path:
      label_lookup_path = os.path.join(
          FLAGS.model_dir, 'imagenet_2012_challenge_label_map_proto.pbtxt')
    if not uid_lookup_path:
      uid_lookup_path = os.path.join(
          FLAGS.model_dir, 'imagenet_synset_to_human_label_map.txt')
    self.node_lookup = self.load(label_lookup_path, uid_lookup_path)

  def load(self, label_lookup_path, uid_lookup_path):
    """Loads a human readable English name for each softmax node.

    Args:
      label_lookup_path: string UID to integer node ID.
      uid_lookup_path: string UID to human-readable string.

    Returns:
      dict from integer node ID to human-readable string.
    """
    if not tf.gfile.Exists(uid_lookup_path):
      tf.logging.fatal('File does not exist %s', uid_lookup_path)
    if not tf.gfile.Exists(label_lookup_path):
      tf.logging.fatal('File does not exist %s', label_lookup_path)

    # Loads mapping from string UID to human-readable string
    proto_as_ascii_lines = tf.gfile.GFile(uid_lookup_path).readlines()
    uid_to_human = {}
    p = re.compile(r'[n\d]*[ \S,]*')
    for line in proto_as_ascii_lines:
      parsed_items = p.findall(line)
      uid = parsed_items[0]
      human_string = parsed_items[2]
      uid_to_human[uid] = human_string

    # Loads mapping from string UID to integer node ID.
    node_id_to_uid = {}
    proto_as_ascii = tf.gfile.GFile(label_lookup_path).readlines()
    for line in proto_as_ascii:
      if line.startswith('  target_class:'):
        target_class = int(line.split(': ')[1])
      if line.startswith('  target_class_string:'):
        target_class_string = line.split(': ')[1]
        node_id_to_uid[target_class] = target_class_string[1:-2]

    # Loads the final mapping of integer node ID to human-readable string
    node_id_to_name = {}
    for key, val in node_id_to_uid.items():
      if val not in uid_to_human:
        tf.logging.fatal('Failed to locate: %s', val)
      name = uid_to_human[val]
      node_id_to_name[key] = name

    return node_id_to_name

  def id_to_string(self, node_id):
    if node_id not in self.node_lookup:
      return ''
    return self.node_lookup[node_id]


def create_graph():
  """Creates a graph from saved GraphDef file and returns a saver."""
  # Creates graph from saved graph_def.pb.
  with tf.gfile.FastGFile(os.path.join(
      FLAGS.model_dir, 'classify_image_graph_def.pb'), 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')


def run_inference_on_image(image):
  """Runs inference on an image.

  Args:
    image: Image file name.

  Returns:
    Nothing
  """

  # Creates graph from saved GraphDef.
  create_graph()

  if not tf.gfile.Exists(image):
    tf.logging.fatal('File does not exist %s', image)
  image_data = tf.gfile.FastGFile(image, 'rb').read()


  with tf.Session() as sess:
    # Some useful tensors:
    #
    # 'softmax:0': A tensor containing the normalized prediction across
    #   1000 labels.
    #
    # 'pool_3:0': A tensor containing the next-to-last layer containing 2048
    #   float description of the image.
    #
    # 'DecodeJpeg/contents:0': A tensor containing a string providing JPEG
    #   encoding of the image.
    #
    # Runs the softmax tensor by feeding the image_data as input to the graph.

    softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
    predictions = sess.run(softmax_tensor,
                           {'DecodeJpeg/contents:0': image_data})
    predictions = np.squeeze(predictions)

    # Creates node ID --> English string lookup.
    node_lookup = NodeLookup()

    top_k = predictions.argsort()[-FLAGS.num_top_predictions:][::-1]

    scores = [] # all scores for this image

    for node_id in top_k:
      human_string = node_lookup.id_to_string(node_id)
      score = predictions[node_id]
      
      # Record the label (human readable string) and the score (confidence),
      # for the most confident labels for each prediction
      scores.append([human_string, score])

      # print('%s (score = %.5f)' % (human_string, score))

    # return all the labels/scores for this image
    return scores


def maybe_download_and_extract():
  """Download and extract model tar file."""
  dest_directory = FLAGS.model_dir
  if not os.path.exists(dest_directory):
    os.makedirs(dest_directory)
  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(dest_directory, filename)
  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (
          filename, float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
  tarfile.open(filepath, 'r:gz').extractall(dest_directory)



def order_images_for_active_learning(directory):
  """takes a directory containing JPGs, and returns the order in which they should be annotated, 
  in order to maximize the accuracy of the trained model as quickly as possible."""
  
  # the image and its scores for each label
  image_scores = []

  # get the current predictions for each image
  images = os.listdir(directory)
  for image in images:
    if image.endswith('jpg'):
      
      with tf.Graph().as_default():
        try:
          scores = run_inference_on_image(directory+'/'+image)

          top_guess = scores[0][0]
          top_score = scores[0][1]

          # TO ADD TO ARRAY OF ALL SCORES:  [IMAGE_NAME, TOP_LABEL, TOP_SCORE, [ALL_SCORES]] 
          image_info = [image, top_guess, top_score, scores]

          print(image_info)  # DEBUGGING: de-comment to see what is recorded for each image 
          image_scores.append(image_info)

        except ValueError:
          # skip files that produce an error 
          sys.stderr.write("Couldn't get a prediction on "+image+" - skipping\n")


  # the final ordered list of images that we will return
  sorted_images = []   

  #BEGIN CODE TO ORDER IMAGES BY IMPORTANCE TO ADD HUMAN LABEL:

  # SORT ARRAY OF ALL SCORES BY 'TOP_SCORE' FOR EACH IMAGE
  image_scores.sort(key=lambda x: x[2], reverse=True)

  for score in image_scores:
    sorted_images.append(score[0])
    print("\n")
    print(score)

  #END CODE TO ORDER IMAGES BY IMPORTANCE TO ADD HUMAN LABEL

  return sorted_images



def main(_):
  maybe_download_and_extract() # download model if not already 

  #check for if a dictionary was passed as an argument
  directory = (FLAGS.directory if FLAGS.directory else
           os.path.join(FLAGS.model_dir, '/'))

  #order the images 
  ordered_images = order_images_for_active_learning(directory)

  #print the optimal order
  print(ordered_images)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  # classify_image_graph_def.pb:
  #   Binary representation of the GraphDef protocol buffer.
  # imagenet_synset_to_human_label_map.txt:
  #   Map from synset ID to a human readable string.
  # imagenet_2012_challenge_label_map_proto.pbtxt:
  #   Text representation of a protocol buffer mapping a label to synset ID.
  parser.add_argument(
      '--model_dir',
      type=str,
      default='/tmp/imagenet',
      help="""\
      Path to classify_image_graph_def.pb,
      imagenet_synset_to_human_label_map.txt, and
      imagenet_2012_challenge_label_map_proto.pbtxt.\
      """
  )
  parser.add_argument(
      '--directory',
      type=str,
      default='',
      help='Absolute path to directory of images.'
  )
  parser.add_argument(
      '--num_top_predictions',
      type=int,
      default=5,
      help='Return this many predictions per image.'
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

