# ============================================
# VGG16 Model for Banana Ripeness Detection
# ============================================

import tensorflow as tf
import numpy as np
import json
import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

print("✅ TensorFlow version:", tf.__version__)

# ------------------------------------------------------
# Dataset paths
# ------------------------------------------------------
train_dir = r"C:\Users\Venkatesh P\Downloads\archive (1)\Banana Ripeness Classification Dataset\train"
val_dir   = r"C:\Users\Venkatesh P\Downloads\archive (1)\Banana Ripeness Classification Dataset\valid"
test_dir  = r"C:\Users\Venkatesh P\Downloads\archive (1)\Banana Ripeness Classification Dataset\test"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 4
EPOCHS = 5

# ------------------------------------------------------
# Data preprocessing
# ------------------------------------------------------
datagen = ImageDataGenerator(rescale=1.0 / 255)

train_gen = datagen.flow_from_directory(
    train_dir, target_size=IMG_SIZE,
    batch_size=BATCH_SIZE, class_mode="categorical"
)

val_gen = datagen.flow_from_directory(
    val_dir, target_size=IMG_SIZE,
    batch_size=BATCH_SIZE, class_mode="categorical"
)

test_gen = datagen.flow_from_directory(
    test_dir, target_size=IMG_SIZE,
    batch_size=BATCH_SIZE, class_mode="categorical",
    shuffle=False
)

with open("class_indices.json", "w") as f:
    json.dump(train_gen.class_indices, f)

# ------------------------------------------------------
# Build VGG16 model
# ------------------------------------------------------
base_model = VGG16(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

base_model.trainable = False  # Freeze convolution layers

x = base_model.output
x = Flatten()(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.5)(x)
output = Dense(NUM_CLASSES, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)

# ------------------------------------------------------
# Compile
# ------------------------------------------------------
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ------------------------------------------------------
# Train
# ------------------------------------------------------
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS
)

# ------------------------------------------------------
# Test
# ------------------------------------------------------
test_loss, test_acc = model.evaluate(test_gen)
print(f"✅ VGG16 Test Accuracy: {test_acc:.4f}")
print(f"✅ VGG16 Test Loss: {test_loss:.4f}")

# ------------------------------------------------------
# Save model
# ------------------------------------------------------
model.save("vgg16_banana.keras")
print("📁 VGG16 model saved as vgg16_banana.keras")

# ------------------------------------------------------
# Plot
# ------------------------------------------------------
os.makedirs("../results", exist_ok=True)

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.plot(history.history["accuracy"], label="Train")
plt.plot(history.history["val_accuracy"], label="Validation")
plt.title("VGG16 Accuracy")
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history["loss"], label="Train")
plt.plot(history.history["val_loss"], label="Validation")
plt.title("VGG16 Loss")
plt.legend()

plt.tight_layout()
plt.savefig("../results/vgg16_training_curves.png")
plt.show()

