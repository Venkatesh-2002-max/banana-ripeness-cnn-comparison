# ============================================
# Banana Ripeness Prediction (LeNet)
# ============================================

import tensorflow as tf
import numpy as np
import json
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image

print("✅ TensorFlow version:", tf.__version__)

# ------------------------------------------------------
# Paths
# ------------------------------------------------------
MODEL_PATH = r"..\models\lenet_banana.keras"
CLASS_INDEX_PATH = r"..\models\class_indices.json"

IMG_SIZE = (32, 32)   # ✅ MUST match LeNet training size

# ------------------------------------------------------
# Load model
# ------------------------------------------------------
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully")

# ------------------------------------------------------
# Load class labels
# ------------------------------------------------------
with open(CLASS_INDEX_PATH, "r") as f:
    class_indices = json.load(f)

idx_to_class = {v: k for k, v in class_indices.items()}
class_labels = [idx_to_class[i] for i in range(len(idx_to_class))]

print("📌 Class labels:", class_labels)

# ------------------------------------------------------
# Image path (CHANGE THIS)
# ------------------------------------------------------
img_path = r"C:\Users\Venkatesh P\Downloads\banana.jpg"
# ------------------------------------------------------
# Load & preprocess image
# ------------------------------------------------------
img = image.load_img(img_path, target_size=IMG_SIZE)
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0   # ✅ SAME as training

# ------------------------------------------------------
# Prediction
# ------------------------------------------------------
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions)
confidence = predictions[0][predicted_class]

# ------------------------------------------------------
# Output
# ------------------------------------------------------
print(f"\n🍌 Prediction: {class_labels[predicted_class]}")
print(f"📊 Confidence: {confidence * 100:.2f}%")

plt.imshow(img)
plt.axis("off")
plt.title(f"{class_labels[predicted_class]} ({confidence*100:.2f}%)")
plt.show()
