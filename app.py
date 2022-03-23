import json
import boto3
import numpy as np
import PIL.Image as Image

import tensorflow as tf

IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256

IMAGE_SHAPE = (IMAGE_WIDTH, IMAGE_HEIGHT)
model = tf.keras.models.load_model('model/gameClass.h5')
s3 = boto3.resource('s3')

def lambda_handler(event, context):
  bucket_name = event['Records'][0]['s3']['bucket']['name']
  key = event['Records'][0]['s3']['object']['key']

  img = readImageFromBucket(key, bucket_name).resize(IMAGE_SHAPE)
  img = np.array(img)/255.0

  prediction = model.predict(img[np.newaxis, ...])
  predicted_class = imagenet_labels[np.argmax(prediction[0], axis=-1)]

  print('ImageName: {0}, Prediction: {1}'.format(key, predicted_class))

def readImageFromBucket(key, bucket_name):
  bucket = s3.Bucket(bucket_name)
  object = bucket.Object(key)
  response = object.get()
  return Image.open(response['Body'])