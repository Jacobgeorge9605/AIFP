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


def classify_new_image(image_path, classifier, categories, threshold=0.5):
    new_image = load_and_preprocess_image(image_path)
    new_image = new_image.reshape(1, -1)
    probabilities = classifier.predict_proba(new_image)[0]
    
    max_probability = max(probabilities)
    print(max_probability)
    
    if max_probability < threshold:
        return "Unknown"  # Return a generic label for unknown classes
    
    prediction = classifier.predict(new_image)[0]
    
    if prediction < len(categories):
        predicted_category = categories[prediction]
    else:
        predicted_category = "Unknown"  # Return "Unknown" for out-of-bounds predictions
    
    return predicted_category
def processing():
    # categories = ["Elephants", "Langurs", "Monkeys", "Nilgai", "Peafowl", "Porcupines", "Spotted_Deer", "Wild_Boars","Humans"]
    categories= [ "Elephants", "Humans", "Wild_Boars" ]

    data = []
    labels = []

    for category in categories:
        try:
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
        except:
            pass
    data = np.array(data).reshape(-1, 150 * 150 * 3)
    labels = np.array(labels)

    model_filename = "animal_classifier_model.pkl"
    if not os.path.isfile(model_filename):
        print("Model file doesn't exist. Training a new model...")
        classifier = train_classifier(data, labels)
        save_classifier(classifier, model_filename)
        print("Model trained and saved.")
    else:
        print("Model file found. Do you want to train the model again? (yes or no)")
        user_input = input()
        
        if user_input.lower() == "yes":
            print("Training a new model...")
            classifier = train_classifier(data, labels)
            save_classifier(classifier, model_filename)
            print("New model trained and saved.")
        else:
            print("Loading existing model...")
            classifier = load_classifier(model_filename)
            print("Model loaded.")

    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", round(accuracy * 100), "%")

    input_directory = "Input"  # Assuming Input is in the same directory as the script
    input_directory = os.path.join(os.path.dirname(__file__), input_directory)
    files = [f for f in os.listdir(input_directory) if os.path.isfile(os.path.join(input_directory, f))]
    files.sort(key=lambda x: os.path.getmtime(os.path.join(input_directory, x)), reverse=True)
    
    if files:
        most_recent_file = files[0]
    else:
        print("No files found in the input directory.")

    try:
        image_path = os.path.join(input_directory, most_recent_file)
        predicted_animal = classify_new_image(image_path, classifier, categories)
        print("Filename:", image_path)
        print("Predicted Animal:", predicted_animal)
    except Exception:
        pass
processing()
