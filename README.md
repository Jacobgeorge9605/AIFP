# AIFP
Official Repo for the AI Wildlife project 2023

# About Our Project
Here firstly PIR Sensor detects the motion of the animal. Once motion is detected, the Servomotor (on top of which camera will be placed) will turn in the drection of detected motion. Camera becomes active and capture the image of the animal. This image will be storedin the folder ./Input. The python file will detect the recently pushed image and it will do ML processing (Algorithm used is RandomForestClassifier of SciKitLearn Library ) and identify the animal. Then corresponding action will be taken like projecting light and sound (Intensities varies from animal to animal). 

# Steps to do
1. Install Following Libraries <br>
	a. NumPY - pip install numpy <br>
	b. Scikit Learn - pip install scikit-learn <br>
