from flask import Flask, render_template, request, redirect, url_for, flash
import os
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)
app.secret_key = "brain_tumor_secret_key"
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Load your pre-trained classification model
classification_model = tf.keras.models.load_model('brain_tumor_classifier_brain_tumor_mri_dataset.h5')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_image(img_path, target_size):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize if required by the model
    return img_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Preprocess and predict classification
        input_shape = classification_model.input_shape[1:3]
        img_array = preprocess_image(filepath, input_shape)
        prediction = classification_model.predict(img_array)
        
        # Get class labels
        class_labels = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
        predicted_class = class_labels[np.argmax(prediction)]
        confidence = np.max(prediction) * 100
        
        # Get confidence scores for all classes
        confidence_scores = {class_labels[i]: f"{prediction[0][i]*100:.2f}%" for i in range(len(class_labels))}
        
        return render_template('result.html', 
                              filename=filename,
                              prediction=predicted_class,
                              confidence=f"{confidence:.2f}%",
                              confidence_scores=confidence_scores)
    
    flash('Invalid file format. Please upload PNG, JPG, or JPEG images.')
    return redirect(request.url)

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)