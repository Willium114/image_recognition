
## 🧩 About the Model

The CNN model was trained on the **CIFAR-10 dataset** with 60,000 32×32 color images belonging to 10 categories:
> airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

### 🔹 Model Architecture
- Conv2D(32 filters, 3×3 kernel, ReLU)
- MaxPooling2D(2×2)
- Conv2D(64 filters, 3×3 kernel, ReLU)
- MaxPooling2D(2×2)
- Conv2D(128 filters, 3×3 kernel, ReLU)
- Flatten()
- Dense(128, ReLU)
- Dropout(0.5)
- Dense(10, Softmax)

The model was compiled using:
```python
optimizer = Adam(learning_rate=0.001)
loss = 'categorical_crossentropy'
metrics = ['accuracy']
git clone https://github.com/Willium114/image_recognition.git
cd image_recognition
streamlit run app.py
🧰 Tools & Libraries Used
Python 3.12

TensorFlow / Keras

Streamlit

NumPy

Pillow (for image handling)
👨‍💻 Author
Developed by: M. Aqib Javed
GitHub: Willium114
