
# 🐱🐶 Custom Image Classifier — Cats & Dogs

This project demonstrates how to build, train, evaluate, and visualize a custom convolutional neural network (CNN) for classifying images of cats and dogs using TensorFlow and Keras.

It includes two main scripts:
- 📈 `Day8_Image_dataset_model.py` — Model training and saving
- 🔍 `custom_model_test.py` — Model testing, evaluation & visualization

---

## 📂 Dataset

The model expects the dataset in the following folder structure:

```
custom_data/
├── train/
│   ├── cat/
│   └── dog/
└── test/
    ├── cat/
    └── dog/
```

Each subfolder (`cat`, `dog`) should contain their respective images.

---

## 🚀 Features

✅ Build and train a CNN from scratch.  
✅ Handles categorical image data using `ImageDataGenerator`.  
✅ Evaluates test accuracy.  
✅ Visualizes predictions with true & predicted labels highlighted (green for correct, red for incorrect).  
✅ Saves the trained model as `custom_model.h5`.  

---

## 🧪 Installation & Setup

### Prerequisites
Make sure you have the following installed:
- Python ≥ 3.7
- TensorFlow ≥ 2.x
- Matplotlib
- NumPy

### Install dependencies
```bash
pip install tensorflow matplotlib numpy
```

### Prepare Dataset
Organize your data as shown in the dataset section above.

---

## 🏗️ Usage

### 1️⃣ Train the Model
Run the training script:
```bash
python Day8_Image_dataset_model.py
```
- Trains the CNN on the `custom_data/train` dataset.
- Validates on the `custom_data/test` dataset.
- Saves the trained model to `custom_model.h5`.

### 2️⃣ Test & Visualize
Run the testing & visualization script:
```bash
python custom_model_test.py
```
- Loads the saved model.
- Evaluates it on the `custom_data/test` dataset.
- Prints the test accuracy.
- Displays a grid of sample predictions with correct & incorrect predictions clearly indicated.

---

## 📜 File Descriptions

### `Day8_Image_dataset_model.py`
- Loads training and testing data with `ImageDataGenerator`.
- Builds a CNN:
  - 2 convolutional layers + max pooling
  - Flatten + Dense layers
  - Softmax output layer
- Trains the model for 10 epochs.
- Saves the trained model.

### `custom_model_test.py`
- Loads the saved model.
- Loads and preprocesses the test set.
- Computes predictions and accuracy.
- Visualizes a batch of predictions with Matplotlib.

---

## 📊 Model Architecture

| Layer Type         | Details                       |
|---------------------|-------------------------------|
| Conv2D              | 32 filters, 3×3, ReLU         |
| MaxPooling2D        | 2×2                           |
| Conv2D              | 64 filters, 3×3, ReLU         |
| MaxPooling2D        | 2×2                           |
| Flatten             | —                             |
| Dense               | 64 units, ReLU                |
| Output Dense        | `num_classes`, Softmax        |

---

## 📈 Sample Output

✅ Example Test Accuracy:
```
Test Accuracy: 92.50%
```

✅ Example Visualization:
- Images displayed in a grid.
- Green title if prediction is correct.
- Red title if prediction is incorrect.

---

## 📖 Notes

- The batch size for training is set to 32, and for testing it is 1 (to predict one image at a time).
- Images are resized to `(64, 64)` for both training and testing.
- You can adjust the number of epochs, layers, and hyperparameters as needed.

---

## 🤝 Contributing

Contributions are welcome! Feel free to fork the repository and submit a pull request with improvements or bug fixes.

---

## 📜 License

This project is open-source and available under the [MIT License](LICENSE).

---

## 🙋‍♂️ Author

**Yash Pal**  
💼 [LinkedIn](https://www.linkedin.com/in/yash-pal-since2004)
