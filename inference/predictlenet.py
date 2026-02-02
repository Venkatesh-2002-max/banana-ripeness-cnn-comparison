import tensorflow as tf
import numpy as np
import json
import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image

# ------------------------------------------------------
# Step 1: Paths
# ------------------------------------------------------
MODEL_PATH = "C:\\Users\\Venkatesh P\\Downloads\\banana-ripeness-cnn-comparison\\models\\lenet_banana.keras"
CLASS_INDEX_PATH = "C:\\Users\\Venkatesh P\\Downloads\\banana-ripeness-cnn-comparison\\models\\class_indices.json"

IMG_SIZE = (32, 32)

# Test image path
IMG_PATH = "C:\\Users\\Venkatesh P\\Downloads\\b3.jpg"

# ------------------------------------------------------
# Step 2: Load model
# ------------------------------------------------------
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ LeNet model loaded")

# ------------------------------------------------------
# Step 3: Load class labels
# ------------------------------------------------------
with open(CLASS_INDEX_PATH, "r") as f:
    class_indices = json.load(f)

# Convert {class: index} → {index: class}
idx_to_class = {v: k for k, v in class_indices.items()}
class_labels = [idx_to_class[i] for i in range(len(idx_to_class))]

print("📌 Class labels:", class_labels)

# ------------------------------------------------------
# Step 4: Load & preprocess image
# ------------------------------------------------------
img = image.load_img(IMG_PATH, target_size=IMG_SIZE)
img_array = image.img_to_array(img)
img_array = img_array / 255.0          # ✅ SAME preprocessing as training
img_array = np.expand_dims(img_array, axis=0)

# ------------------------------------------------------
# Step 5: Prediction
# ------------------------------------------------------
predictions = model.predict(img_array)
predicted_index = np.argmax(predictions[0])
confidence = predictions[0][predicted_index]

# ------------------------------------------------------
# Step 6: Print results
# ------------------------------------------------------
print("\n🔍 Per-class probabilities:")
for i, label in enumerate(class_labels):
    print(f"  [{i}] {label}: {predictions[0][i] * 100:.2f}%")

print(
    f"\n🍌 Final Prediction: {class_labels[predicted_index]} "
    f"({confidence * 100:.2f}%)"
)

# ------------------------------------------------------
# Step 7: Display image
# ------------------------------------------------------
plt.imshow(img)
plt.axis("off")
plt.title(
    f"Prediction: {class_labels[predicted_index]} "
    f"({confidence * 100:.2f}%)"
)
plt.show()
