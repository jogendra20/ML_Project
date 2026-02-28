from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Dummy function to satisfy loader
def f05_score(y_true, y_pred): return 0

model = tf.keras.models.load_model('app/seed_model_v15.h5', custom_objects={'f05_score': f05_score})

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file'].read()
    img = Image.open(io.BytesIO(file)).convert('RGB').resize((224, 224))
    img_array = np.expand_dims(np.array(img), axis=0)
    
    conf = float(model.predict(img_array)[0][0])
    res = "Counterfeit" if conf > 0.5 else "Authentic"
    
    return jsonify({'result': res, 'score': round(conf, 4)})

if __name__ == '__main__':
    app.run(debug=True)