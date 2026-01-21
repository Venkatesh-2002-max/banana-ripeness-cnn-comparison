# ============================================
# LeNet Model for Banana Ripeness Detection
# ============================================

import tensorflow as tf
import numpy as np
import json
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, AveragePooling2D,
    Flatten, Dense, Dropout
)
from tensorflow.keras.optimizers import Adam

print("✅ TensorFlow version:", tf.__version__)

# ------------------------------------------------------
# Step 1: Dataset paths
# ------------------------------------------------------
train_dir = r"C:\Users\Venkatesh P\Downloads\archive (1)\Banana Ripeness Classification Dataset\train"
val_dir   = r"C:\Users\Venkatesh P\Downloads\archive (1)\Banana Ripeness Classification Dataset\valid"
test_dir  = r"C:\Users\Venkatesh P\Downloads\archive (1)\Banana Ripeness Classification Dataset\test"

IMG_SIZE = (32, 32)   # LeNet originally uses 32x32
BATCH_SIZE = 32
NUM_CLASSES = 4       # unripe, ripe, overripe, rotten

# ------------------------------------------------------
# Step 2: Data preprocessing
# ------------------------------------------------------
datagen = ImageDataGenerator(rescale=1.0 / 255)

train_gen = datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

val_gen = datagen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

test_gen = datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

# Save class indices (important for inference & consistency)
with open("class_indices.json", "w") as f:
    json.dump(train_gen.class_indices, f)

print("📌 Class indices:", train_gen.class_indices)

# ------------------------------------------------------
# Step 3: Build LeNet model
# ------------------------------------------------------
model = Sequential([
    Conv2D(6, kernel_size=(5, 5), activation="tanh",
           input_shape=(32, 32, 3)),
    AveragePooling2D(pool_size=(2, 2)),

    Conv2D(16, kernel_size=(5, 5), activation="tanh"),
    AveragePooling2D(pool_size=(2, 2)),

    Flatten(),
    Dense(120, activation="tanh"),
    Dense(84, activation="tanh"),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation="softmax")
])

# ------------------------------------------------------
# Step 4: Compile model
# ------------------------------------------------------
model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ------------------------------------------------------
# Step 5: Train model
# ------------------------------------------------------
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=15
)

# ------------------------------------------------------
# Step 6: Evaluate on test set
# ------------------------------------------------------
test_loss, test_acc = model.evaluate(test_gen)
print(f"✅ LeNet Test Accuracy: {test_acc:.4f}")
print(f"✅ LeNet Test Loss: {test_loss:.4f}")

# ------------------------------------------------------
# Step 7: Save model
# ------------------------------------------------------
model.save("lenet_banana.keras")
print("📁 LeNet model saved as lenet_banana.keras")
