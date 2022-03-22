import json
import boto3
import tensorflow as tf
import numpy as np
import PIL.Image as Image

s3 = boto3.client('s3')

IMG_SIZE = 224
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
