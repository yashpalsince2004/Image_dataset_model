# ✅ Import required libraries
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ✅ Load the saved model
model = keras.models.load_model('custom_model.h5')
print("✅ Model loaded successfully!")

# ✅ Prepare test data
test_datagen = ImageDataGenerator(rescale=1./255)
test_data = test_datagen.flow_from_directory(
    'custom_data/test',
    target_size=(64, 64),
    batch_size=1,  # Predict 1 image at a time
    class_mode='categorical',
    shuffle=False
)

# ✅ Get class labels (cat & dog)
class_indices = test_data.class_indices
labels = list(class_indices.keys())  # ['cat', 'dog']

# ✅ Predict on test data
predictions = model.predict(test_data, verbose=1)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_data.classes

# ✅ Test accuracy
accuracy = np.sum(predicted_classes == true_classes) / len(true_classes)
print(f"✅ Test Accuracy: {accuracy * 100:.2f}%")

# ✅ Visualize Predictions
import math

# ✅ Visualize Predictions — safer version
def visualize_predictions(test_data, predicted_classes, labels, num_images=10):
    # Limit number of images to available predictions
    num_images = min(num_images, len(predicted_classes))

    # Reset generator to ensure correct starting point
    test_data.reset()

    cols = 5
    rows = math.ceil(num_images / cols)

    plt.figure(figsize=(15, 3 * rows))
    for i in range(num_images):
        img_batch, true_label_batch = next(test_data)
        img = img_batch[0]
        true_label = np.argmax(true_label_batch[0])
        pred_label = predicted_classes[i]

        plt.subplot(rows, cols, i + 1)
        plt.imshow(img)
        title_color = 'green' if pred_label == true_label else 'red'
        plt.title(f"True: {labels[true_label]}\nPred: {labels[pred_label]}", color=title_color)
        plt.axis('off')

    plt.tight_layout()
    plt.show()

# ✅ Visualize predictions
visualize_predictions(test_data, predicted_classes, labels, num_images=12)  # Or 20, or all

