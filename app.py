import json
import boto3
import numpy as np
import PIL.Image as Image

import tensorflow as tf

IMAGE_WIDTH = 180
IMAGE_HEIGHT = 180
class_names = ['AC Unity', 'Hitman']
IMAGE_SHAPE = (IMAGE_WIDTH, IMAGE_HEIGHT)
model = tf.keras.models.load_model('model/gameClass.h5')
s3 = boto3.resource('s3')

def lambda_handler(event, context):
	bucket_name = event['Records'][0]['s3']['bucket']['name']
	key = event['Records'][0]['s3']['object']['key']

	img = readImageFromBucket(key, bucket_name).resize(IMAGE_SHAPE)
	# img = np.array(img)/255.0
	img_array = tf.keras.utils.img_to_array(img)
	img_array = tf.expand_dims(img_array, 0)
	
	predictions = model.predict(img_array)
	score = 100 * np.max(tf.nn.softmax(predictions[0]))
	pred = class_names[np.argmax(predictions[0])]
	print(pred, score)


def readImageFromBucket(key, bucket_name):
	bucket = s3.Bucket(bucket_name)
	object = bucket.Object(key)
	response = object.get()
	return Image.open(response['Body'])