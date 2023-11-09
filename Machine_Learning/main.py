from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os
import cv2
import numpy as np
import joblib  

def load_and_preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (150, 150))
    return image

def train_classifier(data, labels):
    classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    classifier.fit(data, labels)
    return classifier

def save_classifier(classifier, model_filename):
    joblib.dump(classifier, model_filename)

def load_classifier(model_filename):
    return joblib.load(model_filename)

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


model_filename = "animal_classifier_model.pkl"
if not os.path.isfile(model_filename):
    classifier = train_classifier(data, labels)
    save_classifier(classifier, model_filename)
else:
    classifier = load_classifier(model_filename)

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", round(accuracy * 100), "%")

image_path = "testModel4.jpg"  
predicted_animal = classify_new_image(image_path, classifier, categories)

print("Filename:", image_path)
print("Predicted Animal:", predicted_animal)
