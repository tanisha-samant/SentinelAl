from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import tempfile
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import librosa
import pickle
import cv2
import subprocess
import requests
import logging
import sys
import base64
import yt_dlp

app = Flask(__name__)
CORS(app)

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s:%(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

emotion_model = None
weapon_model = None
sound_model = None
sound_label_encoder = None

def load_models():
    global emotion_model, weapon_model, sound_model, sound_label_encoder
    try:
        model_dir = "C:/Users/tanis/OneDrive/sentinel/models/"
        emotion_model = load_model(os.path.join(model_dir, "ferplus_cnn.h5"))
        weapon_model = load_model(os.path.join(model_dir, "weapon_detection_model.h5"))
        sound_model = load_model(os.path.join(model_dir, "sentinel_sound_model.h5"))
        with open(os.path.join(model_dir, 'label_encoder.pkl'), 'rb') as f:
            sound_label_encoder = pickle.load(f)
        logging.info("Models loaded successfully")
    except Exception as e:
        logging.error(f"Failed to load models: {e}")
        logging.warning("Continuing with limited functionality")

load_models()

emotion_classes = ['angry', 'contempt', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
weapon_classes = ['gun', 'knife', 'neg']
sound_classes = sound_label_encoder.classes_ if sound_label_encoder else ['aggressive_speech', 'background_noise', 'normal_speech', 'scream']

SAMPLE_RATE = 16000
DURATION = 3.0  # Increased to 3 seconds for more context
SAMPLES = int(SAMPLE_RATE * DURATION)

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'mp4', 'webm', 'wav', 'mp3'}  # Added 'mp3'
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

def send_sms_alert(message):
    logging.info(f"Simulated SMS alert: {message}")
    return True

def preprocess_emotion_image(image_path):
    try:
        img = load_img(image_path, color_mode='grayscale', target_size=(112, 112))
        img_array = img_to_array(img) / 255.0
        logging.debug(f"Emotion image preprocessed: shape={img_array.shape}")
        return np.expand_dims(img_array, axis=0)
    except Exception as e:
        logging.error(f"Error preprocessing emotion image: {e}")
        return None

def preprocess_weapon_image(image_path):
    try:
        img = load_img(image_path, target_size=(224, 224))
        img_array = img_to_array(img) / 255.0
        logging.debug(f"Weapon image preprocessed: shape={img_array.shape}, min={img_array.min()}, max={img_array.max()}")
        return np.expand_dims(img_array, axis=0)
    except Exception as e:
        logging.error(f"Error preprocessing weapon image: {e}")
        return None

def preprocess_audio(audio_data, is_file=True):
    try:
        logging.debug(f"Preprocessing audio, is_file={is_file}, data_size={len(audio_data) if not is_file else 'file'}")
        if is_file:
            logging.debug(f"Loading audio file: {audio_data}")
            audio, sr = librosa.load(audio_data, sr=SAMPLE_RATE)
        else:
            logging.debug("Converting WebM to WAV")
            with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as temp_file:
                temp_file.write(audio_data)
                temp_file_path = temp_file.name
            temp_wav = temp_file_path.replace('.webm', '.wav')
            logging.debug(f"Using ffmpeg to convert {temp_file_path} to {temp_wav}")
            result = subprocess.run(["ffmpeg", "-i", temp_file_path, temp_wav, "-y", "-ac", "1", "-ar", str(SAMPLE_RATE)],
                                   capture_output=True, text=True)
            if result.returncode != 0:
                raise Exception(f"ffmpeg failed: {result.stderr}")
            logging.debug(f"Loading converted WAV: {temp_wav}")
            audio, sr = librosa.load(temp_wav, sr=SAMPLE_RATE)
            os.remove(temp_file_path)
            os.remove(temp_wav)

        # Normalize audio
        audio = audio / np.max(np.abs(audio)) if np.max(np.abs(audio)) > 0 else audio

        logging.debug(f"Audio loaded, length={len(audio)}, sample_rate={sr}")
        if len(audio) > SAMPLES:
            audio = audio[:SAMPLES]
        else:
            audio = np.pad(audio, (0, max(0, SAMPLES - len(audio))), mode='constant')
        logging.debug("Extracting MFCC features")
        mfccs = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=40)
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        logging.error(f"Error preprocessing audio: {e}")
        return None

def analyze_emotion(image_path):
    if not emotion_model:
        return {'error': 'Emotion model not loaded'}
    img_array = preprocess_emotion_image(image_path)
    if img_array is None:
        return {'error': 'Failed to process image for emotion detection'}
    predictions = emotion_model.predict(img_array, batch_size=1)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_class_idx])
    predicted_class = emotion_classes[predicted_class_idx]
    result = {
        'predicted_emotion': predicted_class,
        'confidence': confidence,
        'alert': 'DISTRESS DETECTED!' if predicted_class in ['angry', 'disgust', 'fear', 'sad'] and confidence > 0.9 else None
    }
    logging.debug(f"Emotion detection result: {result}")
    if result['alert']:
        send_sms_alert(f"Distress emotion ({predicted_class}) detected with {confidence:.2%} confidence!")
    return result

def detect_weapons(image_path):
    if not weapon_model:
        return {'error': 'Weapon model not loaded'}
    img_array = preprocess_weapon_image(image_path)
    if img_array is None:
        return {'error': 'Failed to process image for weapon detection'}
    predictions = weapon_model.predict(img_array, batch_size=1)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_class_idx])
    predicted_class = weapon_classes[predicted_class_idx]
    logging.debug(f"Weapon detection raw predictions: {predictions[0]}")
    logging.debug(f"Weapon detection result: class={predicted_class}, confidence={confidence}")
    result = {
        'predicted_weapon': predicted_class,
        'confidence': confidence,
        'alert': 'WEAPON DETECTED!' if predicted_class in ['gun', 'knife'] and confidence > 0.5 else None
    }
    if result['alert']:
        send_sms_alert(f"Weapon ({predicted_class}) detected with {confidence:.2%} confidence!")
    return result

def analyze_audio(audio_data, is_file=True):
    if not sound_model:
        return {'error': 'Sound model not loaded'}
    audio_features = preprocess_audio(audio_data, is_file)
    if audio_features is None:
        return {'error': 'Failed to process audio'}
    features = audio_features.reshape(1, -1)
    predictions = sound_model.predict(features, batch_size=1)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_class_idx])
    predicted_class = sound_classes[predicted_class_idx]
    # Log raw predictions for debugging
    logging.debug(f"Audio raw predictions: {predictions[0]}")
    logging.debug(f"Audio analysis result: class={predicted_class}, confidence={confidence}, probabilities={dict(zip(sound_classes, predictions[0]))}")
    result = {
        'predicted_sound': predicted_class,
        'confidence': confidence,
        'probabilities': {cls: float(prob) for cls, prob in zip(sound_classes, predictions[0])},
        'alert': 'DISTRESS DETECTED!' if predicted_class in ['scream', 'aggressive_speech'] and confidence > 0.7 else None  # Lowered threshold to 0.7
    }
    if result['alert']:
        send_sms_alert(f"Distress sound ({predicted_class}) detected with {confidence:.2%} confidence!")
    return result

def extract_audio_from_video(video_path):
    temp_dir = os.path.dirname(video_path)
    temp_audio = os.path.join(temp_dir, "temp_audio.wav")
    try:
        subprocess.call(["ffmpeg", "-i", video_path, "-q:a", "0", "-map", "a", temp_audio, "-y"],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return temp_audio
    except Exception as e:
        logging.error(f"Error extracting audio: {e}")
        return None

def process_video(video_path):
    if not all([emotion_model, weapon_model]):
        return {'error': 'Required models not loaded'}
    results = {'frames_analyzed': 0, 'emotions': {}, 'weapons': {}, 'timestamps': []}
    try:
        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            return {'error': 'Could not open video file'}
        fps = video.get(cv2.CAP_PROP_FPS)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = max(int(fps) * 10, 1)  # 10-second interval
        logging.debug(f"Processing video: fps={fps}, total_frames={total_frames}, frame_interval={frame_interval}")
        
        temp_audio = extract_audio_from_video(video_path)
        if temp_audio and os.path.exists(temp_audio):
            results['audio'] = analyze_audio(temp_audio)
            os.remove(temp_audio)
        else:
            results['audio'] = {'error': 'Failed to extract audio'}
        
        for frame_idx in range(0, total_frames, frame_interval):
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            success, frame = video.read()
            if not success:
                logging.warning(f"Failed to read frame at index {frame_idx}")
                break
            temp_frame = os.path.join(tempfile.gettempdir(), f"frame_{frame_idx}.jpg")
            cv2.imwrite(temp_frame, frame)
            timestamp = frame_idx / fps if fps > 0 else frame_idx / 30
            logging.debug(f"Analyzing frame at timestamp {timestamp:.2f}s")
            results['emotions'][str(timestamp)] = analyze_emotion(temp_frame)
            results['weapons'][str(timestamp)] = detect_weapons(temp_frame)
            results['frames_analyzed'] += 1
            os.remove(temp_frame)
        
        video.release()
        results['summary'] = summarize_video_analysis(results)
    except Exception as e:
        logging.error(f"Error processing video: {e}")
        return {'error': str(e)}
    return results

def summarize_video_analysis(results):
    summary = {'highest_emotion_alert': None, 'highest_weapon_alert': None, 'audio_alert': results.get('audio', {}).get('alert')}
    for ts, emotion in results.get('emotions', {}).items():
        if emotion.get('alert') and (not summary['highest_emotion_alert'] or emotion.get('confidence', 0) > summary['highest_emotion_alert'].get('confidence', 0)):
            summary['highest_emotion_alert'] = {'timestamp': ts, 'emotion': emotion.get('predicted_emotion'), 'confidence': emotion.get('confidence')}
    for ts, weapon in results.get('weapons', {}).items():
        if weapon.get('alert') and (not summary['highest_weapon_alert'] or weapon.get('confidence', 0) > summary['highest_weapon_alert'].get('confidence', 0)):
            summary['highest_weapon_alert'] = {'timestamp': ts, 'weapon': weapon.get('predicted_weapon'), 'confidence': weapon.get('confidence')}
    threat_level = 'high' if summary['highest_weapon_alert'] or summary['audio_alert'] else 'medium' if summary['highest_emotion_alert'] else 'low'
    summary['threat_level'] = threat_level
    logging.debug(f"Video analysis summary: {summary}")
    return summary

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def download_youtube_video(url):
    try:
        ydl_opts = {
            'format': 'bestvideo[filesize<50M][ext=mp4]+bestaudio[ext=m4a]/best[filesize<50M][ext=mp4]/best',
            'outtmpl': os.path.join(tempfile.gettempdir(), 'youtube_video.%(ext)s'),
            'quiet': True,
            'merge_output_format': 'mp4',
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info).rsplit('.', 1)[0] + '.mp4'
            logging.debug(f"YouTube video downloaded: {filename}")
            return filename
    except yt_dlp.utils.DownloadError as e:
        logging.error(f"Failed to download YouTube video: {e}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error downloading YouTube video: {e}")
        return None

@app.route('/fetch_url', methods=['POST'])
def fetch_url():
    url = request.form.get('url')
    if not url:
        return jsonify({'error': 'No URL provided'}), 400

    try:
        from urllib.parse import urlparse
        domain = urlparse(url).netloc.lower()
        logging.debug(f"Fetching URL: {url}")

        if 'youtube.com' in domain or 'youtu.be' in domain:
            logging.debug("Detected YouTube URL, attempting to download with yt-dlp")
            video_path = download_youtube_video(url)
            if not video_path or not os.path.exists(video_path):
                return jsonify({'error': 'Failed to download YouTube video. It may exceed the 50MB limit, be unavailable, or restricted.'}), 400

            with open(video_path, 'rb') as f:
                content = f.read()
            content_type = 'video/mp4'
            os.remove(video_path)
        else:
            response = requests.get(url, stream=True, timeout=10)
            response.raise_for_status()

            content_length = int(response.headers.get('content-length', 0))
            if content_length > MAX_FILE_SIZE:
                return jsonify({'error': 'Content exceeds 50MB limit'}), 400

            content = b''
            for chunk in response.iter_content(chunk_size=8192):
                content += chunk

            content_type = response.headers.get('content-type', '')
            logging.debug(f"Fetched content type: {content_type}, size: {len(content)} bytes")
            if not (content_type.startswith('image/') or content_type.startswith('video/')):
                return jsonify({'error': f'Unsupported content type: {content_type}. URL must point to an image or video file.'}), 400

        content_base64 = base64.b64encode(content).decode('utf-8')
        return jsonify({'content': content_base64, 'contentType': content_type}), 200
    except requests.RequestException as e:
        logging.error(f"Failed to fetch URL content: {e}")
        return jsonify({'error': f'Network error: {str(e)}'}), 500
    except Exception as e:
        logging.error(f"Unexpected error fetching URL: {e}")
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    file_type = request.form.get('type', 'image')
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': 'Unsupported file type'}), 400
    file.seek(0, os.SEEK_END)
    if file.tell() > MAX_FILE_SIZE:
        return jsonify({'error': 'File exceeds 50MB limit'}), 400
    file.seek(0)
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, file.filename)
    try:
        if file_type == 'audio' and file.filename.endswith('.webm'):
            file_data = file.read()
            results = {'audio': analyze_audio(file_data, is_file=False)}
        elif file_type == 'audio' and file.filename.endswith(('.wav', '.mp3')):  # Added '.mp3'
            file.save(temp_path)
            results = {'audio': analyze_audio(temp_path)}
        else:
            file.save(temp_path)
            results = {'emotion': analyze_emotion(temp_path), 'weapon': detect_weapons(temp_path)} if file_type == 'image' else process_video(temp_path) if file_type == 'video' else {'error': 'Unsupported type'}
    except Exception as e:
        logging.error(f"Analysis error: {e}")
        return jsonify({'error': str(e)}), 500
    finally:
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            os.rmdir(temp_dir)
        except Exception as e:
            logging.error(f"Cleanup error: {e}")
    return jsonify(results), 200

@app.route('/analyze_url', methods=['POST'])
def analyze_url():
    url = request.form.get('url')
    file_type = request.form.get('type', 'image')
    if not url:
        return jsonify({'error': 'No URL provided'}), 400

    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, 'temp_content')
    try:
        from urllib.parse import urlparse
        domain = urlparse(url).netloc.lower()

        if 'youtube.com' in domain or 'youtu.be' in domain:
            logging.debug("Detected YouTube URL, downloading with yt-dlp")
            video_path = download_youtube_video(url)
            if not video_path or not os.path.exists(video_path):
                return jsonify({'error': 'Failed to download YouTube video. It may exceed the 50MB limit, be unavailable, or restricted.'}), 400
            temp_path = video_path
            file_type = 'video'
        else:
            response = requests.get(url, stream=True, timeout=10)
            response.raise_for_status()
            content_length = int(response.headers.get('content-length', 0))
            if content_length > MAX_FILE_SIZE:
                return jsonify({'error': 'Content exceeds 50MB limit'}), 400
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

        results = {'emotion': analyze_emotion(temp_path), 'weapon': detect_weapons(temp_path)} if file_type == 'image' else process_video(temp_path) if file_type == 'video' else {'error': 'Unsupported type'}
    except Exception as e:
        logging.error(f"URL processing error: {e}")
        return jsonify({'error': str(e)}), 500
    finally:
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            os.rmdir(temp_dir)
        except Exception as e:
            logging.error(f"Cleanup error: {e}")
    return jsonify(results), 200

@app.route('/health', methods=['GET'])
def health():
    models_loaded = all(m is not None for m in [emotion_model, weapon_model, sound_model])
    ffmpeg_available = subprocess.run(['ffmpeg', '-version'], capture_output=True).returncode == 0
    disk_space = os.statvfs('/').f_bavail * os.statvfs('/').f_frsize / (1024 * 1024)
    return jsonify({
        'status': 'API is running',
        'models_loaded': models_loaded,
        'ffmpeg_available': ffmpeg_available,
        'disk_space_mb': disk_space,
        'version': '1.0.0'
    }), 200

@app.route('/')
def serve_frontend():
    return send_file('index.html')

if __name__ == '__main__':
    logging.info("Starting SentinelAI application...")
    try:
        app.run(debug=True, host='0.0.0.0', port=5000)
    except Exception as e:
        logging.critical(f"Application failed to start: {e}")
        sys.exit(1)