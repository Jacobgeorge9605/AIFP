# AIFP
Official Repo for the AI Wildlife project 2023

# About Our Project
Here firstly PIR Sensor detects the motion of the animal. Once motion is detected, the Servomotor (on top of which camera will be placed) will turn in the drection of detected motion. Camera becomes active and capture the image of the animal. This image will be storedin the folder ./Input. The python file will detect the recently pushed image and it will do ML processing (Algorithm used is RandomForestClassifier of SciKitLearn Library ) and identify the animal. Then corresponding action will be taken like projecting light and sound (Intensities varies from animal to animal). 

# Steps to do
## Installation & Cloning Repo
1. Install Python
2. Install Following Libraries <br>
	a. NumPY - pip install numpy <br>
	b. Scikit Learn - pip install scikit-learn <br>
 	c. Open CV Python - pip install opencv-python <br>
3. Clone the Repo
   Open Terminal -> "git clone git@github.com:Jacobgeorge9605/AIFP.git" <br>
	[Click here to clone the repository](git@github.com:Jacobgeorge9605/AIFP.git) <br>


## How to add Dataset
1. If you want to detect a new animal, (here animals are represented using classes), inside ./Dataset create a folder with classname (animal name).<br>
2. Inside that folder push all the images related to that class. <br>
3. In the main python file, add the same name (folder name) in the list categories.<br>
4. Done! Now train and Process the model. Now it is good to use !!

# Team Members
We, a team of 4 from LBS College of Engineering, Kasaragod worked on this project.
1. Jacob George (CSE)
2. Pratheek Rao K B (CSE)
3. Nithish Nayak (EC)
4. Akshay (CSE)
