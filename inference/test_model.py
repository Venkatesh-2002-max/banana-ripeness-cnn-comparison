import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import matplotlib.pyplot as plt
import json
import os

# Load the trained model
model = tf.keras.models.load_model("best_mobilenet_banana.h5")

# Load class indices saved during training
class_indices_path = os.path.join(os.path.dirname(__file__), "class_indices.json")

if os.path.exists(class_indices_path):
    with open(class_indices_path, "r") as f:
        class_indices = json.load(f)

    # index -> class name (correct order)
    idx_to_class = {int(v): k for k, v in class_indices.items()}
    class_labels = [idx_to_class[i] for i in range(len(idx_to_class))]
else:
    # ⚠️ Only use if JSON is missing (not recommended)
    class_labels = ["overripe", "ripe", "rotten", "unripe"]

print("Class labels used:", class_labels)

# Path to test image
img_path = r"C:\Users\Venkatesh P\Downloads\archive (1)\Banana Ripeness Classification Dataset\train\rotten\musa-acuminata-unripe-86ed5ab9-1d0a-11ec-83ab-d8c4975e38aa-1-_jpg.rf.abcc310fa09b21de9ab2d34e866de130.jpg"

# Load and preprocess image
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)   # ✅ CORRECT preprocessing

# Prediction
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions[0])
confidence = predictions[0][predicted_class]

# Print probabilities
print("\nPer-class probabilities:")
for i, label in enumerate(class_labels):
    print(f"  [{i}] {label}: {predictions[0][i]*100:.2f}%")

# Display result
plt.imshow(img)
plt.axis("off")
plt.title(f"Prediction: {class_labels[predicted_class]} ({confidence*100:.2f}%)")
plt.show()

