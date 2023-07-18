import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

def predict_image(model, image_path, class_labels):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(64, 64))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the image

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)

    predicted_label = class_labels[predicted_class]
    confidence = prediction[0][predicted_class]

    return img, predicted_label, confidence

# Replace 'image_path' with the actual path to your image file
image_path = "C://Users\AJAY RATHI\Desktop\Flask_Projects\Garbage classification\Garbage classification\metal\metal2.jpg"
class_labels = {0:"cardboard", 1:"glass", 2:"metal", 3:"paper", 4:"plastic", 5:"trash"} # Replace with your actual class labels

model = tf.keras.models.load_model('C://Users\AJAY RATHI\Desktop\Flask_Projects\mymodel.h5')  # Load the trained model

img, predicted_label, confidence = predict_image(model, image_path, class_labels)

# Display the image and its predicted label
plt.imshow(img)
plt.title(f"Predicted Label: {predicted_label} (Confidence: {confidence:.2f})")
plt.axis('off')
plt.show()

# Evaluation
test_generator.reset()  # Reset the generator
true_labels = []
predicted_labels = []

for i in range(len(test_generator)):
    X_test, y_test = test_generator.next()
    true_labels.extend(np.argmax(y_test, axis=1))
    predicted_labels.extend(np.argmax(model.predict(X_test), axis=1))

accuracy = accuracy_score(true_labels, predicted_labels)
print("Accuracy: {:.2f}%".format(accuracy * 100))