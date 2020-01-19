from flask import Flask, request
from image_classifier import create_model, image_transformer, predict_flower

app = Flask(__name__)

