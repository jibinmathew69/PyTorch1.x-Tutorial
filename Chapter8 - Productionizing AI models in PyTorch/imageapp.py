from flask import Flask, request, jsonify
from image_classifier import create_model, predict_flower

app = Flask(__name__)
model = create_model()

@app.route('/predict', methods=['POST'])
def predicted():
    if 'image' not in request.files:
        return jsonify({'error': 'Image not found'}), 400

    image = request.files['image'].read()
    flower_name = predict_flower(model, image)

    return jsonify({'flower_name' : flower_name})

if __name__ == '__main__':
	app.run(debug=True)
