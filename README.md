# Plant Nutrient Detection

This project is a web-based application that detects plant nutrient deficiencies from leaf images using a deep learning model.  
The goal is to help identify nutrient-related stress in plants by analyzing visual patterns in leaf images and highlighting the affected regions.

The application is built using Flask for the backend and PyTorch for model inference.  
Grad-CAM is used to visualize which regions of the image influenced the model’s prediction.

---

## Features

- Upload plant leaf images through a browser
- Predict nutrient deficiency using a trained deep learning model
- Generate Grad-CAM heatmaps for visual explanation
- Display original image and heatmap results
- Simple and clean Flask-based interface

---

## Project Structure

```
plant_nutrient_detection/
│
├── app.py                  # Flask application entry point
├── train_model.py          # Script used to train the model
├── gradcam_utils.py        # Grad-CAM helper functions
│
├── dataset/                # Dataset used for training
├── models/                 # Trained model files (.pth) stored locally
│
├── static/
│   ├── uploads/            # Uploaded images
│   └── results/            # Generated heatmaps
│
├── templates/
│   ├── index.html          # Image upload page
│   └── result.html         # Prediction result page
│
└── README.md
```

---

## Model Details

- Framework: PyTorch  
- Model type: Convolutional Neural Network (CNN)  
- Task: Classification of plant nutrient deficiencies  
- Visualization: Grad-CAM  

The trained model is saved as a `.pth` file and loaded during inference.

Note: The trained model file is not included in this repository to keep the repository size small.

---

## Setup Instructions

### Clone the repository
```
git clone https://github.com/toxx1n/plant_nutrient_detection.git
cd plant_nutrient_detection
```

---

### Create and activate a virtual environment

```
python -m venv venv
```

Windows:
```
venv\Scripts\activate
```

Linux / macOS:
```
source venv/bin/activate
```

---

### Install dependencies

```
pip install flask torch torchvision opencv-python numpy pillow matplotlib
```

---

## Running the Application

```
python app.py
```

Open:
```
http://127.0.0.1:5000
```

---

## Training the Model

```
python train_model.py
```

After training:
- Save the model as a `.pth` file
- Place it inside the `models/` directory
- Update the model path in `app.py` if required

---

## Grad-CAM Visualization

Grad-CAM highlights the regions of the leaf image that most influence the model’s prediction.  
The heatmap is overlaid on the original image and shown in the results page.

---

## Notes

- Model files (`.pth`) are ignored from version control
- Uploaded images and results are stored locally
- This project is intended for educational use

---

## Future Improvements

- Support for additional plant species
- Improved accuracy with larger datasets
- Mobile-friendly interface
- Cloud deployment

---

## Author

Adithya Ranjith  
GitHub: https://github.com/toxx1n

---

If you find this project useful, feel free to star the repository.
