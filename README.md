# SentinelAl
## Overview
SentinelAI is a real-time threat detection system designed to enhance women's safety by analyzing multi-modal inputs (images, videos, and audio) for potential threats. It leverages deep learning models to detect distress emotions, weapons, and alarming sounds, integrated into a Flask-based web application with a user-friendly frontend. The system processes uploaded files, URLs (including YouTube videos), and real-time webcam/microphone inputs, triggering simulated SMS alerts for high-confidence threats.
## Key Features
- **Emotion Detection:** Identifies emotions (e.g., fear, anger) using a custom CNN trained on the FER+ dataset.
- **Weapon Detection:** Detects guns and knives using a MobileNetV2-based model trained on a custom dataset.
- **Sound Analysis:** Classifies audio (e.g., screams, aggressive speech) using a 1D CNN trained on ESC-50 and RAVDESS datasets.
- **Multi-Modal Analysis:** Processes images, videos, and audio, with video frame extraction and audio analysis for comprehensive threat detection.
- **Real-Time Monitoring:** Supports live webcam and microphone input for continuous threat assessment.
- **Web Interface:** Provides a responsive frontend for file uploads, URL analysis, and real-time capture, with visual alerts for threats.
- **Simulated Alerts:** Logs SMS alerts for detected threats (e.g., distress emotions, weapons, or screams with >90% confidence).

## Project Structure
- **app.py:** Flask backend handling API requests, model inference, and media processing.
- **create_label_encoder.py:** Generates a LabelEncoder for audio classes, saved as label_encoder.pkl.
- **index.html:** Frontend interface for user interaction (file uploads, URL input, real-time capture).
- **emotion_detection_model.ipynb:** Jupyter Notebook for training the emotion detection model (ferplus_cnn.h5).
- **weapon_detection_model.ipynb:** Jupyter Notebook for training the weapon detection model (weapon_detection_model.h5).
- **sound_detection_model.ipynb:** Jupyter Notebook for training the audio classification model (sentinel_sound_model.h5).

## Requirements
### System Dependencies
- **Python:** 3.8 or higher
- **FFmpeg:** For audio extraction and conversion (install via apt install ffmpeg on Linux or equivalent for other OS).
- **System Libraries (for pyaudio):** libasound2-dev, portaudio19-dev (install via apt on Linux).

### Python Dependencies
Install the required Python libraries using:
```bash
pip install -r requirements.txt
```
## Setup Instructions

1. Clone the Repository (if applicable):
```bash
git clone <repository-url>
cd sentinelai
```

2.Install Dependencies:
- Install system dependencies:
```bash
sudo apt update
sudo apt install ffmpeg libasound2-dev portaudio19-dev libportaudio2 libportaudiocpp0
```
- Install Python dependencies:
```bash
pip install -r requirements.txt
```
3. Prepare Models and Label Encoder:
- Run ```create_label_encoder.py``` to generate ```label_encoder.pkl```:
```bash
python create_label_encoder.py
```
- Ensure the pre-trained models (ferplus_cnn.h5, weapon_detection_model.h5, sentinel_sound_model.h5) are in the models/ directory.

4. Directory Structure:Ensure the following structure in your project directory:
```plain
sentinelai/
├──notebooks/
│   ├── emotion_detection_model.ipynb
│   ├── weapon_detection_model.ipynb
│   ├── sound_detection_model.ipynb
├── models/
│   ├── ferplus_cnn.h5
│   ├── weapon_detection_model.h5
│   ├── sentinel_sound_model.h5
│   ├── label_encoder.pkl
├── app.py
├── create_label_encoder.py
├── index.html
├── requirements.txt
└── dataset/  # Add datasets for the models
```

5. Run the Application:
```bash
python app.py
```
- The Flask server will start at ```http://localhost:5000```.
- Open a browser and navigate to ```http://localhost:5000``` to access the frontend.



## Usage

1. Frontend Interface (```index.html```):

- **Upload File:** Upload images (JPG, PNG), videos (MP4, WebM), or audio (WAV, MP3) for analysis.
- **URL Analysis:** Enter a YouTube URL or direct image/video URL for processing.
- **Real-Time Capture:** Use webcam and microphone for live threat detection (requires browser permissions).
- Results are displayed for emotion, weapon, and audio analysis, with alerts for detected threats.


2. API Endpoints (```app.py```):

/analyze: POST endpoint for analyzing uploaded files (image, video, audio).
/analyze_url: POST endpoint for analyzing media from URLs.
/health: GET endpoint to check server status and model availability.
/: Serves the index.html frontend.


3. Training Models (if needed):

- Run emotion_detection_model.ipynb with the FER+ dataset to train the emotion detection model.
- Run weapon_detection_model.ipynb with a dataset of gun, knife, and negative images to train the weapon detection model.
- Run sound_detection_model.ipynb in a Colab environment to download ESC-50/RAVDESS datasets and train the audio classification model.



## Datasets

- **Emotion Detection:** FER+ dataset (grayscale images, organized in train, validation, test folders).
- **Weapon Detection:** Custom dataset with gun, knife, and neg subfolders, split into train, val, test by weapon_detection_model.ipynb.
- **Sound Analysis:** ESC-50 (environmental sounds) and RAVDESS (emotional speech), automatically downloaded in sound_detection_model.ipynb.

## Notes

- **Model Training:** The Jupyter Notebooks require datasets to be accessible. Update paths in notebooks if not using the default locations.
- **Real-Time Capture:** Ensure browser permissions for webcam and microphone access. Real-time analysis may be resource-intensive.
- **Limitations:** The SMS alert system is simulated (logs to console) and not integrated with a real messaging service like Twilio.
- **Environment:** Tested in a Python 3.8+ environment with TensorFlow 2.12.0. Use a virtual environment to avoid dependency conflicts.

## Contributing
Contributions are welcome! Please submit issues or pull requests for bug fixes, feature enhancements, or documentation improvements.
## License
This project is licensed under the MIT License.
## Disclaimer
SentinelAI is a research prototype for demonstration purposes. In a real emergency, always contact emergency services.


