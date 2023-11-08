from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os
import cv2
import numpy as np

def load_and_preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (150, 150))
    return image

def classify_new_image(image_path, classifier, categories):
    new_image = load_and_preprocess_image(image_path)
    new_image = new_image.reshape(1, -1)
    prediction = classifier.predict(new_image)[0]
    predicted_category = categories[prediction]
    return predicted_category

categories = ["Elephants", "Langurs", "Monkeys", "Nilgai", "Peafowl", "Porcupines", "Spotted_Deer", "Wild_Boars"]

data = []
labels = []

for category in categories:
    path = os.path.join("Dataset", category)
    class_num = categories.index(category)
    for img in os.listdir(path):
        try:
            img_path = os.path.join(path, img)
            img_array = load_and_preprocess_image(img_path)
            data.append(img_array)
            labels.append(class_num)
        except Exception as e:
            pass

data = np.array(data).reshape(-1, 150 * 150 * 3)
labels = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

image_path = "testModel.jpeg"  # Replace with the path to your image
predicted_animal = classify_new_image(image_path, classifier, categories)
print("Predicted Animal:", predicted_animal)