"Fraud detection application with deep learning" that we prepared for the deep learning course

[![Videoyu İzle](https://img.youtube.com/vi/ePb4xZmYKHo/0.jpg)](https://www.youtube.com/watch?v=ePb4xZmYKHo&t=262s)



Fraud Detection With Deep Learning - Installation Guide
This project aims to detect fraud using deep learning techniques. The project contains code in both Kotlin and Python. Below are step-by-step instructions on how to install and run the project.

Prerequisites
Git
Python 3.8+
pip
Kotlin
Java JDK 8+
Installation Steps
1. Clone the Repository
Use the following command to clone the project to your local machine:


git clone https://github.com/AhmetBugraKaplan/Fraud-Detection-With-Deep-Learning.git
cd Fraud-Detection-With-Deep-Learning
2. Set Up the Python Environment
Create and activate a virtual environment for Python projects:

python3 -m venv env
source env/bin/activate  # macOS/Linux
env\Scripts\activate  # Windows

3. Install Required Python Packages

pip install -r pythoncode/requirements.txt
If requirements.txt is not available, install the packages using the following commands:


pip install numpy pandas scikit-learn tensorflow keras matplotlib seaborn jupyter
4. Prepare the Dataset
Place your dataset in the datasets/ folder:

datasets/
└── FraudDetectedProjectDataSet.xlsx
5. Set Up the Kotlin Project
Build and run the Kotlin project using Gradle:


cd kotlincode/dpdeneme
./gradlew build
./gradlew run
If you are using Windows, use gradlew.bat instead of ./gradlew.



If you need help you can write me on 
Gituhb : https://github.com/AhmetBugraKaplan
Linkedin : https://www.linkedin.com/in/ahmetbugrakaplan/
