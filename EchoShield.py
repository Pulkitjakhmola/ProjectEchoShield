from flask import Flask, render_template, request, jsonify, send_file, flash, redirect, url_for
import numpy as np
import librosa
import soundfile as sf
from scipy.signal import butter, filtfilt
import hashlib 
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
import io
import base64
from werkzeug.utils import secure_filename
import tempfile
from pathlib import Path

app = Flask(__name__)
app.secret_key = 'echo_shield_secret_key_2024'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['UPLOAD_FOLDER'] = 'Uploads'

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

class EchoShield:
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
        self.fingerprint_freq_range = (18000, 22000)  # High frequency range
        self.fingerprint_duration = 0.5  # Duration of fingerprint pattern in seconds
        self.fingerprint_interval = 10.0  # Interval between fingerprints in seconds
        self.db_file = "fingerprint_db.json"
        
    def generate_fingerprint(self, lecture_id, instructor_name=None):
        """Generate a unique fingerprint for a lecture"""
        # Create unique identifier
        timestamp = datetime.now().isoformat()
        if instructor_name:
            unique_string = f"{lecture_id}_{instructor_name}_{timestamp}"
        else:
            unique_string = f"{lecture_id}_{timestamp}"
        
        # Generate hash for reproducible randomness
        fingerprint_hash = hashlib.sha256(unique_string.encode()).hexdigest()
        
        # Convert hash to frequency pattern
        freq_pattern = []
        for i in range(0, len(fingerprint_hash), 4):
            # Convert hex chunk to frequency
            hex_chunk = fingerprint_hash[i:i+4]
            freq_offset = int(hex_chunk, 16) % 1000  # 0-999 Hz offset
            freq = self.fingerprint_freq_range[0] + freq_offset
            freq_pattern.append(freq)
        
        fingerprint_data = {
            'lecture_id': lecture_id,
            'instructor_name': instructor_name,
            'timestamp': timestamp,
            'hash': fingerprint_hash,
            'frequencies': freq_pattern[:8]  # Use first 8 frequencies
        }
        
        return fingerprint_data
    
    def create_fingerprint_audio(self, fingerprint_data, duration=0.5):
        """Create audio signal from fingerprint data"""
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        signal = np.zeros_like(t)
        
        # Create multi-frequency pattern
        for i, freq in enumerate(fingerprint_data['frequencies']):
            # Each frequency has different amplitude and phase
            amplitude = 0.01 * (0.5 + 0.5 * np.sin(i))  # Very low amplitude
            phase = i * np.pi / 4  # Different phase for each frequency
            signal += amplitude * np.sin(2 * np.pi * freq * t + phase)
        
        # Apply envelope to make it less detectable
        envelope = np.exp(-t * 2)  # Exponential decay
        signal *= envelope
        
        return signal
    
    def generate_protection_audio(self, fingerprint_data, protection_duration=60.0):
        """Generate standalone protection audio to play alongside lectures"""
        # Calculate number of fingerprint insertions
        num_fingerprints = int(protection_duration / self.fingerprint_interval)
        
        # Generate base noise
        protection_signal = np.random.normal(0, 0.005, 
                                           int(self.sample_rate * protection_duration))
        
        # Apply high-pass filter to make noise inaudible
        nyquist = self.sample_rate / 2
        high_freq = 17000 / nyquist
        b, a = butter(4, high_freq, btype='high')
        protection_signal = filtfilt(b, a, protection_signal)
        
        # Insert fingerprints at regular intervals
        fingerprint_audio = self.create_fingerprint_audio(fingerprint_data)
        fingerprint_samples = len(fingerprint_audio)
        interval_samples = int(self.fingerprint_interval * self.sample_rate)
        
        for i in range(num_fingerprints):
            start_idx = i * interval_samples
            end_idx = start_idx + fingerprint_samples
            
            if end_idx <= len(protection_signal):
                protection_signal[start_idx:end_idx] += fingerprint_audio
        
        return protection_signal
    
    def mix_with_lecture(self, lecture_audio_path, fingerprint_data, output_path):
        """Mix protection signal directly into lecture audio"""
        # Load lecture audio
        lecture_audio, sr = librosa.load(lecture_audio_path, sr=self.sample_rate)
        lecture_duration = len(lecture_audio) / self.sample_rate
        
        # Generate protection signal for full lecture duration
        protection_signal = self.generate_protection_audio(fingerprint_data, lecture_duration)
        
        # Mix protection with lecture (very low level)
        mixed_audio = lecture_audio + protection_signal[:len(lecture_audio)] * 0.1
        
        # Ensure no clipping
        mixed_audio = np.clip(mixed_audio, -0.95, 0.95)
        
        # Save mixed audio
        sf.write(output_path, mixed_audio, self.sample_rate)
        
        return mixed_audio
    
    def extract_high_freq_content(self, audio_signal):
        """Extract high frequency content from audio"""
        # Apply high-pass filter
        nyquist = self.sample_rate / 2
        high_freq = 17000 / nyquist
        b, a = butter(4, high_freq, btype='high')
        high_freq_signal = filtfilt(b, a, audio_signal)
        
        return high_freq_signal
    
    def detect_fingerprint_frequencies(self, audio_signal, known_fingerprint=None):
        """Detect fingerprint frequencies in audio signal"""
        # Extract high frequency content
        high_freq_signal = self.extract_high_freq_content(audio_signal)
        
        # Perform FFT analysis in chunks
        chunk_duration = 2.0  # seconds
        chunk_samples = int(chunk_duration * self.sample_rate)
        num_chunks = len(high_freq_signal) // chunk_samples
        
        detected_frequencies = []
        confidences = []
        
        for i in range(num_chunks):
            start_idx = i * chunk_samples
            end_idx = start_idx + chunk_samples
            chunk = high_freq_signal[start_idx:end_idx]
            
            # FFT analysis
            fft = np.fft.fft(chunk)
            freqs = np.fft.fftfreq(len(chunk), 1/self.sample_rate)
            magnitude = np.abs(fft)
            
            # Focus on fingerprint frequency range
            freq_mask = (freqs >= self.fingerprint_freq_range[0]) & \
                       (freqs <= self.fingerprint_freq_range[1])
            
            if np.any(freq_mask):
                chunk_freqs = freqs[freq_mask]
                chunk_magnitudes = magnitude[freq_mask]
                
                # Find peaks
                if len(chunk_magnitudes) > 0:
                    threshold = np.mean(chunk_magnitudes) + 2 * np.std(chunk_magnitudes)
                    peaks = chunk_magnitudes > threshold
                    
                    if np.any(peaks):
                        peak_freqs = chunk_freqs[peaks]
                        peak_mags = chunk_magnitudes[peaks]
                        
                        for freq, mag in zip(peak_freqs, peak_mags):
                            detected_frequencies.append(freq)
                            confidences.append(mag)
        
        return detected_frequencies, confidences
    
    def analyze_recording(self, audio_path, known_fingerprints=None):
        """Analyze audio recording for Echo Shield protection"""
        # Load audio
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)
        
        # Detect frequencies
        detected_freqs, confidences = self.detect_fingerprint_frequencies(audio)
        
        results = {
            'file_path': audio_path,
            'protection_detected': len(detected_freqs) > 0,
            'detected_frequencies': detected_freqs[:20],  # Limit output
            'avg_confidence': np.mean(confidences) if confidences else 0,
            'num_detections': len(detected_freqs)
        }
        
        # If known fingerprints provided, try to match
        if known_fingerprints:
            best_match = None
            best_score = 0
            
            for fp_id, fp_data in known_fingerprints.items():
                if 'frequencies' in fp_data:
                    # Calculate match score
                    score = self.calculate_match_score(detected_freqs, fp_data['frequencies'])
                    if score > best_score:
                        best_score = score
                        best_match = fp_id
            
            results['best_match'] = best_match
            results['match_score'] = best_score
            results['likely_match'] = best_score > 0.3  # Threshold for positive match
        
        return results
    
    def calculate_match_score(self, detected_freqs, fingerprint_freqs, tolerance=50):
        """Calculate how well detected frequencies match fingerprint"""
        if not detected_freqs or not fingerprint_freqs:
            return 0
        
        matches = 0
        for fp_freq in fingerprint_freqs:
            for det_freq in detected_freqs:
                if abs(det_freq - fp_freq) <= tolerance:
                    matches += 1
                    break
        
        return matches / len(fingerprint_freqs)
    
    def save_fingerprint_database(self, fingerprints, filepath=None):
        """Save fingerprint database to file"""
        if filepath is None:
            filepath = self.db_file
        with open(filepath, 'w') as f:
            json.dump(fingerprints, f, indent=2)
    
    def load_fingerprint_database(self, filepath=None):
        """Load fingerprint database from file"""
        if filepath is None:
            filepath = self.db_file
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                return json.load(f)
        return {}
    
    def plot_frequency_analysis(self, audio_path):
        """Plot frequency analysis of audio file and return base64 encoded image"""
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)
        
        # Compute spectrogram
        D = librosa.stft(audio)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        
        plt.figure(figsize=(12, 8))
        
        # Plot full spectrogram
        plt.subplot(2, 1, 1)
        librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Full Frequency Spectrogram')
        plt.ylim(0, sr/2)
        
        # Plot high frequency region
        plt.subplot(2, 1, 2)
        librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz')
        plt.colorbar(format='%+2.0f dB')
        plt.title('High Frequency Region (15kHz+)')
        plt.ylim(15000, sr/2)
        
        plt.tight_layout()
        
        # Convert plot to base64 string
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        plot_data = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        return plot_data

# Initialize Echo Shield
echo_shield = EchoShield()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate')
def generate_page():
    return render_template('generate.html')

@app.route('/api/generate', methods=['POST'])
def generate_protection():
    try:
        data = request.get_json()
        lecture_id = data.get('lecture_id')
        instructor_name = data.get('instructor_name')
        duration = float(data.get('duration', 60))
        
        if not lecture_id:
            return jsonify({'success': False, 'error': 'Lecture ID is required'})
        
        # Generate fingerprint and protection audio
        fingerprint = echo_shield.generate_fingerprint(lecture_id, instructor_name)
        protection_audio = echo_shield.generate_protection_audio(fingerprint, duration * 60)
        
        # Save to temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav', 
                                               dir=app.config['UPLOAD_FOLDER'])
        sf.write(temp_file.name, protection_audio, echo_shield.sample_rate)
        
        # Save fingerprint to database
        fingerprint_db = echo_shield.load_fingerprint_database()
        fingerprint_db[lecture_id] = fingerprint
        echo_shield.save_fingerprint_database(fingerprint_db)
        
        return jsonify({
            'success': True,
            'download_url': f'/download/{os.path.basename(temp_file.name)}',
            'fingerprint': {
                'lecture_id': lecture_id,
                'instructor_name': instructor_name,
                'frequencies': fingerprint['frequencies'][:3],
                'timestamp': fingerprint['timestamp']
            }
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/mix')
def mix_page():
    return render_template('mix.html')

@app.route('/api/mix', methods=['POST'])
def mix_protection():
    try:
        if 'audio_file' not in request.files:
            return jsonify({'success': False, 'error': 'No audio file uploaded'})
        
        file = request.files['audio_file']
        lecture_id = request.form.get('lecture_id')
        instructor_name = request.form.get('instructor_name')
        
        if not lecture_id:
            return jsonify({'success': False, 'error': 'Lecture ID is required'})
        
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'})
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(input_path)
        
        # Generate fingerprint
        fingerprint = echo_shield.generate_fingerprint(lecture_id, instructor_name)
        
        # Mix protection with lecture
        output_filename = f"{os.path.splitext(filename)[0]}_protected.wav"
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
        
        echo_shield.mix_with_lecture(input_path, fingerprint, output_path)
        
        # Save fingerprint to database
        fingerprint_db = echo_shield.load_fingerprint_database()
        fingerprint_db[lecture_id] = fingerprint
        echo_shield.save_fingerprint_database(fingerprint_db)
        
        # Clean up input file
        os.remove(input_path)
        
        return jsonify({
            'success': True,
            'download_url': f'/download/{output_filename}',
            'fingerprint': {
                'lecture_id': lecture_id,
                'instructor_name': instructor_name,
                'frequencies': fingerprint['frequencies'][:3],
                'timestamp': fingerprint['timestamp']
            }
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/analyze')
def analyze_page():
    return render_template('analyze.html')

@app.route('/api/analyze', methods=['POST'])
def analyze_recording():
    try:
        if 'audio_file' not in request.files:
            return jsonify({'success': False, 'error': 'No audio file uploaded'})
        
        file = request.files['audio_file']
        
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'})
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(input_path)
        
        # Load fingerprint database
        fingerprint_db = echo_shield.load_fingerprint_database()
        
        # Analyze recording
        results = echo_shield.analyze_recording(input_path, fingerprint_db)
        
        # Generate frequency plot
        plot_data = echo_shield.plot_frequency_analysis(input_path)
        
        # Clean up uploaded file
        os.remove(input_path)
        
        # Prepare response
        response_data = {
            'success': True,
            'protection_detected': results['protection_detected'],
            'num_detections': results['num_detections'],
            'avg_confidence': round(results['avg_confidence'], 3),
            'plot_data': plot_data
        }
        
        if 'best_match' in results and results['best_match']:
            match_info = fingerprint_db[results['best_match']]
            response_data['match'] = {
                'lecture_id': results['best_match'],
                'instructor_name': match_info.get('instructor_name', 'N/A'),
                'match_score': round(results['match_score'], 3),
                'timestamp': match_info['timestamp'],
                'confidence_level': 'HIGH' if results['match_score'] > 0.5 else 'MEDIUM' if results['match_score'] > 0.3 else 'LOW'
            }
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/database')
def database_page():
    fingerprint_db = echo_shield.load_fingerprint_database()
    return render_template('database.html', fingerprints=fingerprint_db)

@app.route('/api/database/delete', methods=['POST'])
def delete_fingerprint():
    try:
        data = request.get_json()
        lecture_id = data.get('lecture_id')
        
        fingerprint_db = echo_shield.load_fingerprint_database()
        if lecture_id in fingerprint_db:
            del fingerprint_db[lecture_id]
            echo_shield.save_fingerprint_database(fingerprint_db)
            return jsonify({'success': True})
        else:
            return jsonify({'success': False, 'error': 'Lecture ID not found'})
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/database')
def get_database():
    fingerprint_db = echo_shield.load_fingerprint_database()
    return jsonify(fingerprint_db)

@app.route('/download/<filename>')
def download_file(filename):
    try:
        return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename), 
                        as_attachment=True)
    except Exception as e:
        flash(f'Error downloading file: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/api/database/export')
def export_database():
    try:
        fingerprint_db = echo_shield.load_fingerprint_database()
        
        # Create temporary export file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.json', 
                                               dir=app.config['UPLOAD_FOLDER'])
        with open(temp_file.name, 'w') as f:
            json.dump(fingerprint_db, f, indent=2)
        
        return send_file(temp_file.name, as_attachment=True, 
                        download_name='fingerprint_export.json')
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)