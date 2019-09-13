
#Research SLIIT 072 / Pedestrian count

import tensorflow as tf

# Object detection imports
from utils import backbone #CNN is composed of two blocks sharing a backbone. This in turn aims to produce discriminative features that will be used to estimate candidate objects as well as to predict the class of those objects
from api import pedestrians_counting


input_video = "./input_images_and_videos/dataset1_kirulapone.avi" #kirulapona video1
#input_video = "./input_images_and_videos/pedestrians_walking_.mp4"
#input_video = "./input_images_and_videos/zenital22.avi" #walking video



# By default I use an "SSD with Mobilenet" model here. See the detection model zoo (https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.
detection_graph, category_index = backbone.set_model('ssd_mobilenet_v1_coco_2018_01_28')
targeted_objects = "person" # (for counting targeted objects)
is_color_recognition_enabled = 1

pedestrians_counting.pedestrians(input_video, detection_graph, category_index, is_color_recognition_enabled, targeted_objects) # targeted objects counting

