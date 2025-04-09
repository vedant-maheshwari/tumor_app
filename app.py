from flask import Flask, render_template, request, redirect, url_for, flash
import os
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2

app = Flask(__name__)
app.secret_key = "brain_tumor_secret_key"
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Load your pre-trained classification model
classification_model = tf.keras.models.load_model('brain_tumor_classifier_brain_tumor_mri_dataset.h5')

# Load your pre-trained U-Net segmentation model
segmentation_model = tf.keras.models.load_model('model_best_checkpoint.h5', compile=False)

# Print the model's input shape
print("Classification model input shape:", classification_model.input_shape)
print("Segmentation model input shape:", segmentation_model.input_shape)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_image(img_path, target_size):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize if required by the model
    return img_array

def segment_tumor_unet(img_path, input_shape):
    # Read the image and convert to grayscale
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(gray, (input_shape[1], input_shape[0]))
    img_array = img_resized.astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension
    img_array = np.expand_dims(img_array, axis=0)   # Add batch dimension
    
    # Predict segmentation mask using U-Net
    mask = segmentation_model.predict(img_array)
    mask = (mask > 0.5).astype(np.uint8)
    mask = mask.reshape((input_shape[0], input_shape[1]))
    
    # Calculate tumor size
    tumor_size = np.sum(mask == 1)
    total_pixels = mask.size
    tumor_percentage = (tumor_size / total_pixels) * 100
    
    return mask, tumor_percentage

def apply_mask(img_path, mask, input_shape):
    # Read the original image
    img = cv2.imread(img_path)
    img_resized = cv2.resize(img, (input_shape[1], input_shape[0]))
    
    # Apply the mask to the original image
    masked_img = cv2.bitwise_and(img_resized, img_resized, mask=mask)
    return masked_img

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
        classification_input_shape = classification_model.input_shape[1:3]
        img_array = preprocess_image(filepath, classification_input_shape)
        classification_prediction = classification_model.predict(img_array)
        
        # Get classification results
        class_labels = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
        predicted_class = class_labels[np.argmax(classification_prediction)]
        confidence = np.max(classification_prediction) * 100
        confidence_scores = {class_labels[i]: f"{classification_prediction[0][i]*100:.2f}%" for i in range(len(class_labels))}
        
        # Segment tumor and estimate size if tumor is detected
        tumor_detected = predicted_class != 'No Tumor'
        masked_filename = None
        tumor_size = None
        
        if tumor_detected:
            segmentation_input_shape = segmentation_model.input_shape[1:3]
            mask, tumor_size = segment_tumor_unet(filepath, segmentation_input_shape)
            masked_img = apply_mask(filepath, mask, segmentation_input_shape)
            masked_filename = f"masked_{filename}"
            masked_filepath = os.path.join(app.config['UPLOAD_FOLDER'], masked_filename)
            cv2.imwrite(masked_filepath, masked_img)
        
        return render_template('result.html', 
                              filename=filename,
                              masked_filename=masked_filename,
                              prediction=predicted_class,
                              confidence=f"{confidence:.2f}%",
                              confidence_scores=confidence_scores,
                              tumor_size=tumor_size,
                              tumor_detected=tumor_detected)
    
    flash('Invalid file format. Please upload PNG, JPG, or JPEG images.')
    return redirect(request.url)

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)