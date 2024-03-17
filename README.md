# MoodSwing-Emotion-Classifier

This MoodSwing-Emotion-Classifier is a Flask-based web application that leverages a TensorFlow model trained on the "Emotion Dataset" from Kaggle. It classifies user-input text comments into emotions such as joy, anger, and fear, demonstrating the integration of a TensorFlow machine learning model with a web frontend for real-time emotion classification.

## Features

- **Real-time Emotion Classification**: Users can enter a text comment, and the application predicts the emotion behind the text.
- **TensorFlow Integration**: Utilizes a model built and trained using TensorFlow, showcasing how deep learning models can be seamlessly integrated into web applications.
- **Flask Backend**: A lightweight Flask application serves as the backend, handling the preprocessing of text inputs and model inference.

## Getting Started

These instructions will guide you through setting up the project on your local machine for development and testing purposes.

### Prerequisites

Before you begin, ensure you have the following installed:

- Python 3.6 or higher
- pip and virtualenv

### Installation

1. **Clone the repository**:
   
```bash
git clone https://github.com/txhno/MoodSwing-Emotion-Classifier.git
```

2. **Navigate to the project directory:**:

```bash
cd MoodSwing-Emotion-Classifier
```

3. **Create a virtual environment and activate it:**:

```bash
python3 -m venv venv
source venv/bin/activate
```

4. **Install the required packages:**:

```bash
pip install -r requirements.txt
```

### Training the Model

Before launching the Flask application, you need to train the model using the "Emotion Dataset" from Kaggle. This step is crucial for preparing the model for emotion classification.

1. **Download the dataset** [emotion-dataset by abdallahwagih](https://www.kaggle.com/datasets/abdallahwagih/emotion-dataset) and place `Emotion_classify_Data.csv` in the root directory of the project.

2. **Ensure you are in the project's root directory**.

3. **Run the training script**:

```bash
python model/model_training.py
```
This script trains the model and saves it, making it ready for inference by the Flask app.

### Usage
1. **Run the Flask application**:

```bash
python run.py
```
2. **Open a web browser and navigate to `http://localhost:5000` to access the application.**
3. **Use the text input to enter a comment and submit it to see the model’s prediction of the emotion.**

## Built With

- **TensorFlow** - The machine learning framework used for building and training the emotion classification model.
- **Flask** - The web framework used for the application backend.
- **HTML/CSS/JavaScript** - Used for the frontend interface.

## Authors

- **Roshan Warrier** - *Project Owner* - [txhno](https://github.com/txhno)
