# ============================================
# Banana Ripeness Prediction using VGG16
# ============================================

import tensorflow as tf
import numpy as np
import json
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image

print("✅ TensorFlow version:", tf.__version__)

# ------------------------------------------------------
# Paths (EDIT ONLY IF NEEDED)
# ------------------------------------------------------
MODEL_PATH = r"..\models\vgg16_banana.keras"
CLASS_INDEX_PATH = r"..\models\class_indices.json"

IMG_SIZE = (224, 224)   # ✅ VGG16 input size

# ------------------------------------------------------
# Load trained VGG16 model
# ------------------------------------------------------
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ VGG16 model loaded")

# ------------------------------------------------------
# Load class labels
# ------------------------------------------------------
with open(CLASS_INDEX_PATH, "r") as f:
    class_indices = json.load(f)

idx_to_class = {v: k for k, v in class_indices.items()}
class_labels = [idx_to_class[i] for i in range(len(idx_to_class))]

print("📌 Classes:", class_labels)

# ------------------------------------------------------
# Image to predict (CHANGE THIS)
# ------------------------------------------------------
img_path = r"C:\Users\Venkatesh P\Downloads\ban.jpg"

# ------------------------------------------------------
# Load & preprocess image
# ------------------------------------------------------
img = image.load_img(img_path, target_size=IMG_SIZE)
img_array = image.img_to_array(img)
img_array = img_array / 255.0     # same as training
img_array = np.expand_dims(img_array, axis=0)

# ------------------------------------------------------
# Prediction
# ------------------------------------------------------
predictions = model.predict(img_array)
predicted_index = np.argmax(predictions)
confidence = predictions[0][predicted_index] * 100

predicted_class = class_labels[predicted_index]

# ------------------------------------------------------
# Output
# ------------------------------------------------------
print("\n🔍 Prediction Results")
for i, label in enumerate(class_labels):
    print(f"{label:10s}: {predictions[0][i]*100:.2f}%")

print(f"\n✅ Final Prediction: {predicted_class}")
print(f"🎯 Confidence: {confidence:.2f}%")

plt.imshow(img)
plt.axis("off")
plt.title(f"{predicted_class} ({confidence:.2f}%)")
plt.show()
