import cv2
import time
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os
import cv2
import numpy as np
import joblib  
import pygame


class RIM_Project:

    def load_classifier(model_filename):
        return joblib.load(model_filename)
    
    def load_and_preprocess_image(image_path):
        image = cv2.imread(image_path)
        image = cv2.resize(image, (150, 150))
        return image
    
    def classify_new_image(image_path, classifier, categories, threshold=0.52):
        new_image = RIM_Project.load_and_preprocess_image(image_path)
        new_image = new_image.reshape(1, -1)
        probabilities = classifier.predict_proba(new_image)[0]
        
        
        max_probability = max(probabilities)
        # print(max_probability)
        prediction = classifier.predict(new_image)[0]
        predicted_category = categories[prediction]
        # if max_probability < threshold:
        #     return f"Other, {max_probability}% {predicted_category}"
        
        return predicted_category, f" {max_probability}"
    
    def predictImage():
        categories = ["Elephants", "Langurs", "Monkeys", "Nilgai", "Peafowl", "Porcupines", "Spotted_Deer", "Wild_Boars"]
        model_filename = "MachineLearning/animal_classifier_model.pkl"
        
        classifier = RIM_Project.load_classifier(model_filename)

        input_directory = "MachineLearning/images"  # Assuming Input is in the same directory as the script
        input_directory = os.path.join(os.path.dirname(__file__), input_directory)
        files = [f for f in os.listdir(input_directory) if os.path.isfile(os.path.join(input_directory, f))]
        files.sort(key=lambda x: os.path.getmtime(os.path.join(input_directory, x)), reverse=True)
        
        if files:
            most_recent_file = files[0]
        else:
            return "No files found in the input directory."

        try:
            image_path = os.path.join(input_directory, most_recent_file)
            predicted_animal = RIM_Project.classify_new_image(image_path, classifier, categories)
            return predicted_animal
        except Exception:
            pass
    
    @staticmethod
    def play_music(music_filename, duration=6):
        try:
            pygame.mixer.init()
            pygame.mixer.music.load(music_filename)
            pygame.mixer.music.play()
            time.sleep(duration)
        except:
            pass
        
        pygame.mixer.music.stop()

    def startProject():
        cap = cv2.VideoCapture(0)
        output_folder = "MachineLearning/images"
        os.makedirs(output_folder, exist_ok=True)

        start_time = time.time()

        try:
            while True:
                ret, frame = cap.read()
                cv2.imshow('Video Capture', frame)
                cv2.waitKey(1)  
                current_time = time.time()
                if current_time - start_time >= 20:
                    start_time = current_time  
                    image_name = f"{output_folder}/image_{int(current_time)}.png"
                    cv2.imwrite(image_name, frame)
                    print(f"Frame captured and saved: {image_name}")
                    # SEND this image to the ML Model


                    result = RIM_Project.predictImage()
                    print(result)

                    music_filename = "MachineLearning/music/ELEPHANT - Sound Effect.mp3"
                    try:
                        RIM_Project.play_music(music_filename)
                    except:
                        pass


                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Terminated Successfully! ")
                    break

        except KeyboardInterrupt:
            print("Video capture terminated by user.")

        finally:
            cap.release()
            cv2.destroyAllWindows()



RIM_Project.startProject()


    