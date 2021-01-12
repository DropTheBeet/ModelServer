from flask import Flask, request, jsonify
from flask.json import JSONEncoder

import io, sys, time

import tensorflow as tf
import requests
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import numpy as np
from PIL import Image

from tensorflow.python.eager import def_function
from tensorflow.python.framework import tensor_spec
from tensorflow.python.util import nest

FLAGS(sys.argv)

flags.DEFINE_string('weights', './checkpoints/yolov3.tf',
                    'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_string('output', './serving/yolov3/1', 'path to saved_model')
flags.DEFINE_string('classes', './data/coco.names', 'path to classes file')
flags.DEFINE_string('image', './data/girl.png', 'path to input image')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')



app = Flask(__name__)

model = tf.saved_model.load('/home/bit/Yolo_coco_cpu_test/serving/yolo_coco_cpu_test/1')
infer = model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
logging.info(infer.structured_outputs)

def transform_images(x_train, size):
    x_train = tf.image.resize(x_train, (size, size))
    x_train = x_train / 255
    return x_train

class CustomJSONEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)

        return JSONEncoder.default(self, obj)


@app.route('/')
def home():

    return 'Hello, World!'

@app.route('/recognized_tag', methods=['POST'])
def download_img():
    url = request.json['img_url']
    response = requests.get(url, stream=True)

    try:
        # class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
        class_names = [c.strip() for c in open('/home/bit/Yolo_coco_cpu_test/data/coco.names').readlines()]
        logging.info('classes loaded')

        img = tf.image.decode_image(io.BytesIO(response.raw.read()).getvalue(), channels=3)
        img = tf.expand_dims(img, 0)
        img = transform_images(img, 416)

        t1 = time.time()
        outputs = infer(img)
        boxes, scores, classes, nums = outputs["yolo_nms"], outputs[
            "yolo_nms_1"], outputs["yolo_nms_2"], outputs["yolo_nms_3"]
        t2 = time.time()
        logging.info('time: {}'.format(t2 - t1))


        logging.info('detections:')
        tag_result = []
        for i in range(nums[0]):
            tag_result.append({int(classes[0][i]): boxes[0][i].numpy().tolist()})
            
    except Exception as e:
        print("error")
        print(e)
        return jsonify({
            'error' : e
        })


    return jsonify({ 'tag': tag_result })



if __name__ == '__main__':
    try:
        app.run(debug=True)
    except SystemExit:
        pass