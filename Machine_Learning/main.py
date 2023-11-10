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

def classify_new_image(image_path, classifier, categories, threshold=0.7):
    new_image = load_and_preprocess_image(image_path)
    new_image = new_image.reshape(1, -1)
    probabilities = classifier.predict_proba(new_image)[0]
    
    max_probability = max(probabilities)
    if max_probability < threshold:
        return "Other"
    
    prediction = classifier.predict(new_image)[0]
    predicted_category = categories[prediction]
    return predicted_category

def processing():
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
    input_directory = "./Input"
    files = [f for f in os.listdir(input_directory) if os.path.isfile(os.path.join(input_directory, f))]
    files.sort(key=lambda x: os.path.getmtime(os.path.join(input_directory, x)), reverse=True)
    if files:
        most_recent_file = files[0]
        # print("Most recently added file:", most_recent_file)
    else:
        print("No files found in the input directory.")

    image_path = f"Input/{most_recent_file}"
    predicted_animal = classify_new_image(image_path, classifier, categories)
    print("Filename:", image_path)
    print("Predicted Animal:", predicted_animal)

    # image_path = "testModel4.jpg"  
    # while True:
    #     image_path = input("Enter the relative path of the image to test (or 'exit' to quit): ")
        
    #     if image_path.lower() == 'exit':
    #         break 
        
    #     predicted_animal = classify_new_image(image_path, classifier, categories)
        
    #     print("Filename:", image_path)
    #     print("Predicted Animal:", predicted_animal)
        
    #     test_again = input("Do you want to test another image? (yes or no): ")
        
    #     if test_again.lower() != 'yes':
    #         break  
processing()