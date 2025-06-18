#!/usr/bin/env python3
"""
Echo Shield Desktop Application
Audio Enhancement System for Online Learning
Run this during Zoom calls for enhanced audio quality
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import sqlite3
import hashlib
import json
import numpy as np
import pyaudio
import threading
import time
from datetime import datetime
import requests
import os
import wave
import librosa
from scipy import signal
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class DatabaseManager:
    def __init__(self, db_file="echo_shield.db"):
        # Use absolute path to ensure all instances use same database
        self.db_file = os.path.abspath(db_file)
        self.init_database()
    
    def init_database(self):
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                user_type TEXT NOT NULL,
                full_name TEXT,
                student_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS active_sessions (
                id INTEGER PRIMARY KEY,
                student_id TEXT NOT NULL,
                student_name TEXT NOT NULL,
                frequencies TEXT NOT NULL,
                start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_ping TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status TEXT DEFAULT 'active'
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS frequency_database (
                id INTEGER PRIMARY KEY,
                student_id TEXT NOT NULL,
                student_name TEXT NOT NULL,
                frequencies TEXT NOT NULL,
                session_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                session_duration INTEGER,
                UNIQUE(student_id, session_date)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analysis_reports (
                id INTEGER PRIMARY KEY,
                file_name TEXT NOT NULL,
                analyzed_by TEXT NOT NULL,
                detection_results TEXT NOT NULL,
                confidence_score REAL,
                analysis_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def create_user(self, email, password, user_type, full_name, student_id=None):
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        
        try:
            cursor.execute('''
                INSERT INTO users (email, password_hash, user_type, full_name, student_id)
                VALUES (?, ?, ?, ?, ?)
            ''', (email, password_hash, user_type, full_name, student_id))
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False
        finally:
            conn.close()
    
    def verify_user(self, email, password):
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        cursor.execute('''
            SELECT user_type, full_name, student_id FROM users 
            WHERE email = ? AND password_hash = ?
        ''', (email, password_hash))
        result = cursor.fetchone()
        conn.close()
        return result
    
    def get_active_sessions(self):
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT student_id, student_name, frequencies, start_time, last_ping
            FROM active_sessions WHERE status = 'active'
            ORDER BY start_time DESC
        ''')
        results = cursor.fetchall()
        conn.close()
        
        sessions = []
        for row in results:
            student_id, student_name, frequencies, start_time, last_ping = row
            sessions.append({
                'student_id': student_id,
                'student_name': student_name,
                'frequencies': json.loads(frequencies),
                'start_time': start_time,
                'duration': int((datetime.now() - datetime.fromisoformat(start_time)).total_seconds())
            })
        return sessions
    
    def save_frequency_record(self, student_id, student_name, frequencies, duration):
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        try:
            cursor.execute('''
                INSERT OR REPLACE INTO frequency_database 
                (student_id, student_name, frequencies, session_duration)
                VALUES (?, ?, ?, ?)
            ''', (student_id, student_name, json.dumps(frequencies), duration))
            conn.commit()
        finally:
            conn.close()
    
    def get_all_frequency_records(self):
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT student_id, student_name, frequencies, session_date, session_duration
            FROM frequency_database ORDER BY session_date DESC
        ''')
        results = cursor.fetchall()
        conn.close()
        return results
    
    def save_analysis_report(self, file_name, analyzed_by, detection_results, confidence_score):
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO analysis_reports 
            (file_name, analyzed_by, detection_results, confidence_score)
            VALUES (?, ?, ?, ?)
        ''', (file_name, analyzed_by, json.dumps(detection_results), confidence_score))
        conn.commit()
        conn.close()

class AudioAnalyzer:
    def __init__(self):
        self.sample_rate = 44100
        self.freq_tolerance = 50  # Hz tolerance for frequency detection
    
    def load_audio_file(self, file_path):
        """Load audio file and return audio data and sample rate"""
        try:
            if file_path.lower().endswith('.wav'):
                audio_data, sr = librosa.load(file_path, sr=self.sample_rate)
            else:
                audio_data, sr = librosa.load(file_path, sr=self.sample_rate)
            return audio_data, sr
        except Exception as e:
            raise Exception(f"Error loading audio file: {str(e)}")
    
    def detect_frequencies(self, audio_data, target_frequencies):
        """Detect if target frequencies are present in audio"""
        # Compute FFT
        fft_data = np.abs(fft(audio_data))
        freqs = fftfreq(len(audio_data), 1/self.sample_rate)
        
        # Only consider positive frequencies
        positive_freq_idx = freqs > 0
        freqs = freqs[positive_freq_idx]
        fft_data = fft_data[positive_freq_idx]
        
        detection_results = {}
        confidence_scores = []
        
        for target_freq in target_frequencies:
            # Find the closest frequency bin
            freq_idx = np.argmin(np.abs(freqs - target_freq))
            closest_freq = freqs[freq_idx]
            
            # Check if within tolerance
            if abs(closest_freq - target_freq) <= self.freq_tolerance:
                # Calculate confidence based on amplitude
                amplitude = fft_data[freq_idx]
                
                # Normalize amplitude (simple approach)
                max_amplitude = np.max(fft_data)
                confidence = min(amplitude / max_amplitude * 100, 100)
                
                detection_results[target_freq] = {
                    'detected': True,
                    'detected_freq': closest_freq,
                    'confidence': confidence,
                    'amplitude': amplitude
                }
                confidence_scores.append(confidence)
            else:
                detection_results[target_freq] = {
                    'detected': False,
                    'detected_freq': None,
                    'confidence': 0,
                    'amplitude': 0
                }
                confidence_scores.append(0)
        
        # Overall confidence is average of individual confidences
        overall_confidence = np.mean(confidence_scores) if confidence_scores else 0
        
        return detection_results, overall_confidence
    
    def analyze_spectrogram(self, audio_data, target_frequencies):
        """Generate spectrogram analysis"""
        # Compute spectrogram
        f, t, Sxx = signal.spectrogram(audio_data, self.sample_rate, nperseg=1024)
        
        # Find target frequencies in spectrogram
        freq_indices = []
        for target_freq in target_frequencies:
            freq_idx = np.argmin(np.abs(f - target_freq))
            freq_indices.append(freq_idx)
        
        return f, t, Sxx, freq_indices

class AudioEnhancer:
    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.is_running = False
        self.frequencies = []
        
    def generate_frequencies(self, student_name, student_id):
        """Generate unique inaudible frequencies for student"""
        seed_string = f"{student_name.upper().strip()}_{student_id}"
        seed_hash = hashlib.sha256(seed_string.encode()).hexdigest()
        
        frequencies = []
        base_freq = 17000  # Inaudible to most humans
        
        for i in range(4):
            hash_segment = seed_hash[i*4:(i+1)*4]
            freq_offset = int(hash_segment, 16) % 3000
            frequency = base_freq + (i * 500) + (freq_offset / 10)
            frequencies.append(round(frequency, 2))
        
        return frequencies
    
    def audio_callback(self, in_data, frame_count, time_info, status):
        """Audio callback to inject enhancement frequencies"""
        if not self.is_running:
            return (np.zeros(frame_count, dtype=np.float32).tobytes(), pyaudio.paContinue)
        
        # Generate enhancement signal
        sample_rate = 44100
        t = np.linspace(0, frame_count/sample_rate, frame_count)
        enhancement = np.zeros(frame_count)
        
        for freq in self.frequencies:
            enhancement += 0.01 * np.sin(2 * np.pi * freq * t)  # Very low amplitude
        
        return (enhancement.astype(np.float32).tobytes(), pyaudio.paContinue)
    
    def start_enhancement(self, frequencies):
        """Start audio enhancement"""
        self.frequencies = frequencies
        self.is_running = True
        
        try:
            self.stream = self.p.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=44100,
                output=True,
                frames_per_buffer=1024,
                stream_callback=self.audio_callback
            )
            self.stream.start_stream()
            return True
        except Exception as e:
            print(f"Audio error: {e}")
            return False
    
    def stop_enhancement(self):
        """Stop audio enhancement"""
        self.is_running = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
    
    def cleanup(self):
        self.stop_enhancement()
        self.p.terminate()

class EchoShieldApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Echo Shield - Audio Enhancement System")
        self.root.geometry("900x700")
        self.root.configure(bg='#2c3e50')
        
        self.db = DatabaseManager()
        self.audio_enhancer = AudioEnhancer()
        self.audio_analyzer = AudioAnalyzer()
        self.current_user = None
        self.session_active = False
        self.session_start_time = None
        
        self.create_login_interface()
    
    def create_login_interface(self):
        # Clear window
        for widget in self.root.winfo_children():
            widget.destroy()
        
        # Main frame
        main_frame = tk.Frame(self.root, bg='#2c3e50')
        main_frame.pack(expand=True, fill='both', padx=20, pady=20)
        
        # Title
        title = tk.Label(main_frame, text="Echo Shield", font=('Arial', 24, 'bold'), 
                        fg='#3498db', bg='#2c3e50')
        title.pack(pady=20)
        
        subtitle = tk.Label(main_frame, text="Audio Enhancement for Online Learning", 
                           font=('Arial', 12), fg='#ecf0f1', bg='#2c3e50')
        subtitle.pack(pady=(0, 30))
        
        # Notebook for tabs
        notebook = ttk.Notebook(main_frame)
        notebook.pack(expand=True, fill='both')
        
        # Login tab
        login_frame = tk.Frame(notebook, bg='#34495e')
        notebook.add(login_frame, text="Login")
        
        self.create_login_form(login_frame)
        
        # Register tab
        register_frame = tk.Frame(notebook, bg='#34495e')
        notebook.add(register_frame, text="Register")
        
        self.create_register_form(register_frame)
    
    def create_login_form(self, parent):
        form_frame = tk.Frame(parent, bg='#34495e')
        form_frame.pack(expand=True)
        
        tk.Label(form_frame, text="Email:", font=('Arial', 12), fg='#ecf0f1', bg='#34495e').pack(pady=5)
        self.login_email = tk.Entry(form_frame, font=('Arial', 12), width=30)
        self.login_email.pack(pady=5)
        
        tk.Label(form_frame, text="Password:", font=('Arial', 12), fg='#ecf0f1', bg='#34495e').pack(pady=5)
        self.login_password = tk.Entry(form_frame, font=('Arial', 12), width=30, show='*')
        self.login_password.pack(pady=5)
        
        tk.Button(form_frame, text="Login", font=('Arial', 12), bg='#3498db', fg='white',
                 command=self.login, width=20).pack(pady=20)
    
    def create_register_form(self, parent):
        form_frame = tk.Frame(parent, bg='#34495e')
        form_frame.pack(expand=True)
        
        tk.Label(form_frame, text="Email:", font=('Arial', 12), fg='#ecf0f1', bg='#34495e').pack(pady=5)
        self.reg_email = tk.Entry(form_frame, font=('Arial', 12), width=30)
        self.reg_email.pack(pady=5)
        
        tk.Label(form_frame, text="Password:", font=('Arial', 12), fg='#ecf0f1', bg='#34495e').pack(pady=5)
        self.reg_password = tk.Entry(form_frame, font=('Arial', 12), width=30, show='*')
        self.reg_password.pack(pady=5)
        
        tk.Label(form_frame, text="Full Name:", font=('Arial', 12), fg='#ecf0f1', bg='#34495e').pack(pady=5)
        self.reg_name = tk.Entry(form_frame, font=('Arial', 12), width=30)
        self.reg_name.pack(pady=5)
        
        tk.Label(form_frame, text="User Type:", font=('Arial', 12), fg='#ecf0f1', bg='#34495e').pack(pady=5)
        self.user_type = ttk.Combobox(form_frame, values=["student", "faculty"], width=27)
        self.user_type.pack(pady=5)
        self.user_type.bind('<<ComboboxSelected>>', self.on_user_type_change)
        
        self.student_id_label = tk.Label(form_frame, text="Student ID:", font=('Arial', 12), fg='#ecf0f1', bg='#34495e')
        self.reg_student_id = tk.Entry(form_frame, font=('Arial', 12), width=30)
        
        tk.Button(form_frame, text="Register", font=('Arial', 12), bg='#27ae60', fg='white',
                 command=self.register, width=20).pack(pady=20)
    
    def on_user_type_change(self, event):
        if self.user_type.get() == "student":
            self.student_id_label.pack(pady=5)
            self.reg_student_id.pack(pady=5)
        else:
            self.student_id_label.pack_forget()
            self.reg_student_id.pack_forget()
    
    def login(self):
        email = self.login_email.get()
        password = self.login_password.get()
        
        if not email or not password:
            messagebox.showerror("Error", "Please fill all fields")
            return
        
        result = self.db.verify_user(email, password)
        if result:
            user_type, full_name, student_id = result
            self.current_user = {
                'email': email,
                'user_type': user_type,
                'full_name': full_name,
                'student_id': student_id
            }
            
            if user_type == 'student':
                self.create_student_dashboard()
            else:
                self.create_faculty_dashboard()
        else:
            messagebox.showerror("Error", "Invalid credentials")
    
    def register(self):
        email = self.reg_email.get()
        password = self.reg_password.get()
        full_name = self.reg_name.get()
        user_type = self.user_type.get()
        student_id = self.reg_student_id.get() if user_type == "student" else None
        
        if not all([email, password, full_name, user_type]):
            messagebox.showerror("Error", "Please fill all required fields")
            return
        
        if user_type == "student" and not student_id:
            messagebox.showerror("Error", "Student ID required for student accounts")
            return
        
        success = self.db.create_user(email, password, user_type, full_name, student_id)
        if success:
            messagebox.showinfo("Success", "Account created successfully")
            # Clear form
            self.reg_email.delete(0, tk.END)
            self.reg_password.delete(0, tk.END)
            self.reg_name.delete(0, tk.END)
            self.reg_student_id.delete(0, tk.END)
        else:
            messagebox.showerror("Error", "Email already exists")
    
    def create_student_dashboard(self):
        for widget in self.root.winfo_children():
            widget.destroy()
        
        main_frame = tk.Frame(self.root, bg='#2c3e50')
        main_frame.pack(expand=True, fill='both', padx=20, pady=20)
        
        # Header
        header = tk.Frame(main_frame, bg='#2c3e50')
        header.pack(fill='x', pady=(0, 20))
        
        tk.Label(header, text=f"Welcome, {self.current_user['full_name']}", 
                font=('Arial', 18, 'bold'), fg='#3498db', bg='#2c3e50').pack(side='left')
        
        tk.Button(header, text="Logout", font=('Arial', 10), bg='#e74c3c', fg='white',
                 command=self.logout).pack(side='right')
        
        # Main content
        content = tk.Frame(main_frame, bg='#34495e', relief='ridge', bd=2)
        content.pack(expand=True, fill='both', padx=10, pady=10)
        
        tk.Label(content, text="Audio Enhancement System", font=('Arial', 16, 'bold'), 
                fg='#ecf0f1', bg='#34495e').pack(pady=20)
        
        tk.Label(content, text="Enhance your audio quality during Zoom calls", 
                font=('Arial', 12), fg='#bdc3c7', bg='#34495e').pack(pady=10)
        
        # Status
        self.status_label = tk.Label(content, text="Status: Ready", font=('Arial', 12, 'bold'), 
                                   fg='#f39c12', bg='#34495e')
        self.status_label.pack(pady=20)
        
        # Control buttons
        button_frame = tk.Frame(content, bg='#34495e')
        button_frame.pack(pady=20)
        
        self.start_btn = tk.Button(button_frame, text="Start Enhancement", font=('Arial', 14), 
                                  bg='#27ae60', fg='white', width=20, command=self.start_enhancement)
        self.start_btn.pack(pady=10)
        
        self.stop_btn = tk.Button(button_frame, text="Stop Enhancement", font=('Arial', 14), 
                                 bg='#e74c3c', fg='white', width=20, command=self.stop_enhancement, state='disabled')
        self.stop_btn.pack(pady=10)
        
        # Info
        info_text = "This system enhances audio clarity and reduces background noise during online classes."
        tk.Label(content, text=info_text, font=('Arial', 10), fg='#95a5a6', bg='#34495e', 
                wraplength=600).pack(pady=20)
    
    def create_faculty_dashboard(self):
        for widget in self.root.winfo_children():
            widget.destroy()
        
        main_frame = tk.Frame(self.root, bg='#2c3e50')
        main_frame.pack(expand=True, fill='both', padx=20, pady=20)
        
        # Header
        header = tk.Frame(main_frame, bg='#2c3e50')
        header.pack(fill='x', pady=(0, 20))
        
        tk.Label(header, text=f"Faculty Dashboard - {self.current_user['full_name']}", 
                font=('Arial', 18, 'bold'), fg='#3498db', bg='#2c3e50').pack(side='left')
        
        tk.Button(header, text="Logout", font=('Arial', 10), bg='#e74c3c', fg='white',
                 command=self.logout).pack(side='right')
        
        # Create notebook for different views
        self.faculty_notebook = ttk.Notebook(main_frame)
        self.faculty_notebook.pack(expand=True, fill='both')
        
        # Active Sessions Tab
        sessions_frame = tk.Frame(self.faculty_notebook, bg='#34495e')
        self.faculty_notebook.add(sessions_frame, text="Active Sessions")
        self.create_sessions_tab(sessions_frame)
        
        # Recording Analysis Tab
        analysis_frame = tk.Frame(self.faculty_notebook, bg='#34495e')
        self.faculty_notebook.add(analysis_frame, text="Recording Analysis")
        self.create_analysis_tab(analysis_frame)
        
        # Frequency Database Tab
        database_frame = tk.Frame(self.faculty_notebook, bg='#34495e')
        self.faculty_notebook.add(database_frame, text="Frequency Database")
        self.create_database_tab(database_frame)
    
    def create_sessions_tab(self, parent):
        # Control buttons
        control_frame = tk.Frame(parent, bg='#34495e')
        control_frame.pack(fill='x', pady=10)
        
        tk.Button(control_frame, text="Refresh Now", font=('Arial', 10), bg='#3498db', fg='white',
                 command=self.refresh_sessions).pack(side='left', padx=5)
        
        tk.Button(control_frame, text="Auto-Refresh", font=('Arial', 10), bg='#27ae60', fg='white',
                 command=self.toggle_auto_refresh).pack(side='left', padx=5)
        
        # Status info
        self.session_count_label = tk.Label(control_frame, text="Active Sessions: 0", 
                                          font=('Arial', 12, 'bold'), fg='#e74c3c', bg='#34495e')
        self.session_count_label.pack(side='right')
        
        # Sessions list
        tk.Label(parent, text="Active Enhancement Sessions", font=('Arial', 16, 'bold'), 
                fg='#ecf0f1', bg='#34495e').pack(pady=(10, 5))
        
        # Treeview for sessions
        columns = ('Student ID', 'Student Name', 'Frequencies', 'Duration', 'Last Ping')
        self.sessions_tree = ttk.Treeview(parent, columns=columns, show='headings', height=15)
        
        # Configure columns
        column_widths = {'Student ID': 100, 'Student Name': 150, 'Frequencies': 300, 'Duration': 100, 'Last Ping': 150}
        for col in columns:
            self.sessions_tree.heading(col, text=col)
            self.sessions_tree.column(col, width=column_widths.get(col, 120))
        
        # Scrollbars
        v_scrollbar = ttk.Scrollbar(parent, orient='vertical', command=self.sessions_tree.yview)
        self.sessions_tree.configure(yscroll=v_scrollbar.set)
        
        # Pack treeview and scrollbars
        self.sessions_tree.pack(side='left', expand=True, fill='both', padx=(10, 0), pady=10)
        v_scrollbar.pack(side='right', fill='y', pady=10)
        
        # Initialize auto-refresh
        self.auto_refresh_active = False
        self.refresh_sessions()
        self.toggle_auto_refresh()
    
    def create_analysis_tab(self, parent):
        # Title
        tk.Label(parent, text="Audio Recording Analysis", font=('Arial', 16, 'bold'), 
                fg='#ecf0f1', bg='#34495e').pack(pady=10)
        
        # File selection
        file_frame = tk.Frame(parent, bg='#34495e')
        file_frame.pack(fill='x', padx=20, pady=10)
        
        tk.Label(file_frame, text="Select Audio File:", font=('Arial', 12), 
                fg='#ecf0f1', bg='#34495e').pack(side='left')
        
        self.selected_file_label = tk.Label(file_frame, text="No file selected", 
                                        font=('Arial', 10), fg='#95a5a6', bg='#34495e')
        self.selected_file_label.pack(side='left', padx=10)
        
        tk.Button(file_frame, text="Browse", font=('Arial', 10), bg='#3498db', fg='white',
                command=self.browse_audio_file).pack(side='right')
        
        # Analyze button
        tk.Button(parent, text="Analyze Recording", font=('Arial', 14), bg='#e67e22', fg='white',
                command=self.analyze_recording, width=20).pack(pady=20)
        
        # Results frame
        results_frame = tk.Frame(parent, bg='#34495e', relief='ridge', bd=2)
        results_frame.pack(expand=True, fill='both', padx=20, pady=10)
        
        tk.Label(results_frame, text="Analysis Results", font=('Arial', 14, 'bold'), 
                fg='#ecf0f1', bg='#34495e').pack(pady=10)
        
        # Results text area
        self.results_text = tk.Text(results_frame, height=15, bg='#2c3e50', fg='#ecf0f1', 
                                font=('Courier', 10))
        results_scrollbar = ttk.Scrollbar(results_frame, orient='vertical', command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=results_scrollbar.set)
        
        self.results_text.pack(side='left', expand=True, fill='both', padx=10, pady=10)
        results_scrollbar.pack(side='right', fill='y', pady=10)
        
        # Initialize
        self.selected_audio_file = None
    
    def create_database_tab(self, parent):
        # Title
        tk.Label(parent, text="Frequency Database", font=('Arial', 16, 'bold'), 
                fg='#ecf0f1', bg='#34495e').pack(pady=10)
        
        # Refresh button
        tk.Button(parent, text="Refresh Database", font=('Arial', 12), bg='#3498db', fg='white',
                 command=self.refresh_frequency_database).pack(pady=10)
        
        # Database treeview
        columns = ('Student ID', 'Student Name', 'Frequencies', 'Session Date', 'Duration')
        self.db_tree = ttk.Treeview(parent, columns=columns, show='headings', height=20)
        
        # Configure columns
        for col in columns:
            self.db_tree.heading(col, text=col)
            self.db_tree.column(col, width=150)
        
        # Scrollbar
        db_scrollbar = ttk.Scrollbar(parent, orient='vertical', command=self.db_tree.yview)
        self.db_tree.configure(yscrollcommand=db_scrollbar.set)
        
        # Pack database view
        self.db_tree.pack(side='left', expand=True, fill='both', padx=(10, 0), pady=10)
        db_scrollbar.pack(side='right', fill='y', pady=10)
        
        # Initialize database view
        self.refresh_frequency_database()
    
    def browse_audio_file(self):
        """Browse for audio file to analyze"""
        file_types = [
            ("Audio Files", "*.wav *.mp3 *.m4a *.flac *.ogg"),
            ("WAV Files", "*.wav"),
            ("MP3 Files", "*.mp3"),
            ("All Files", "*.*")
        ]
        
        file_path = filedialog.askopenfilename(
            title="Select Audio File for Analysis",
            filetypes=file_types
        )
        
        if file_path:
            self.selected_audio_file = file_path
            filename = os.path.basename(file_path)
            self.selected_file_label.config(text=f"Selected: {filename}", fg='#27ae60')
    
    def refresh_student_list(self):
        """Refresh the list of students for frequency matching"""
        records = self.db.get_all_frequency_records()
        student_options = []
        
        for record in records:
            student_id, student_name, _, _, _ = record
            option = f"{student_name} ({student_id})"
            if option not in student_options:
                student_options.append(option)
        
        self.student_combo['values'] = student_options
        if student_options:
            self.student_combo.set(student_options[0])
    
    def analyze_recording(self):
        """Analyze the selected recording for frequency signatures and match against database"""
        if not self.selected_audio_file:
            messagebox.showerror("Error", "Please select an audio file first")
            return
        
        try:
            # Clear previous results
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, "Analyzing audio file...\n\n")
            self.results_text.update()
            
            # Load and analyze audio
            audio_data, sample_rate = self.audio_analyzer.load_audio_file(self.selected_audio_file)
            
            self.results_text.insert(tk.END, f"File: {os.path.basename(self.selected_audio_file)}\n")
            self.results_text.insert(tk.END, f"Sample Rate: {sample_rate} Hz\n")
            self.results_text.insert(tk.END, f"Duration: {len(audio_data)/sample_rate:.2f} seconds\n\n")
            self.results_text.update()
            
            # Get all frequency records from the database
            records = self.db.get_all_frequency_records()
            best_match = None
            best_confidence = 0
            target_frequencies = None
            student_name = None
            student_id = None
            
            for record in records:
                current_student_id, current_student_name, frequencies_str, _, _ = record
                current_frequencies = json.loads(frequencies_str)
                
                # Detect frequencies in the audio
                detection_results, overall_confidence = self.audio_analyzer.detect_frequencies(
                    audio_data, current_frequencies
                )
                
                # Count detected frequencies
                detected_count = sum(1 for result in detection_results.values() if result['detected'])
                
                # Update best match if confidence is higher and at least 2 frequencies are detected
                if overall_confidence > best_confidence and detected_count >= 2:
                    best_confidence = overall_confidence
                    best_match = detection_results
                    target_frequencies = current_frequencies
                    student_name = current_student_name
                    student_id = current_student_id
            
            # Display results
            self.results_text.insert(tk.END, "=== FREQUENCY DETECTION RESULTS ===\n\n")
            
            if best_match:
                detected_count = sum(1 for result in best_match.values() if result['detected'])
                for freq, result in best_match.items():
                    if result['detected']:
                        status = "✓ DETECTED"
                        color = "green"
                    else:
                        status = "✗ NOT DETECTED"
                        color = "red"
                    
                    self.results_text.insert(tk.END, 
                        f"Target: {freq:.1f} Hz - {status}\n"
                        f"  Confidence: {result['confidence']:.1f}%\n"
                        f"  Detected at: {result['detected_freq']:.1f} Hz\n\n"
                    )
                
                # Overall assessment
                self.results_text.insert(tk.END, "=== OVERALL ASSESSMENT ===\n\n")
                self.results_text.insert(tk.END, f"Frequencies Detected: {detected_count}/{len(target_frequencies)}\n")
                self.results_text.insert(tk.END, f"Overall Confidence: {best_confidence:.1f}%\n")
                self.results_text.insert(tk.END, f"Matched Student: {student_name} (ID: {student_id})\n\n")
                
                if detected_count >= 3:
                    verdict = "RECORDING LIKELY PROTECTED"
                    self.results_text.insert(tk.END, f"VERDICT: {verdict}\n")
                    self.results_text.insert(tk.END, "This recording contains expected frequency signatures.\n")
                elif detected_count >= 2:
                    verdict = "PARTIAL DETECTION"
                    self.results_text.insert(tk.END, f"VERDICT: {verdict}\n")
                    self.results_text.insert(tk.END, "Some frequency signatures detected. Recording may be protected.\n")
                else:
                    verdict = "NO PROTECTION DETECTED"
                    self.results_text.insert(tk.END, f"VERDICT: {verdict}\n")
                    self.results_text.insert(tk.END, "Recording does not contain expected frequency signatures.\n")
            else:
                self.results_text.insert(tk.END, "No matching frequency signatures detected in the database.\n")
                self.results_text.insert(tk.END, "VERDICT: NO PROTECTION DETECTED\n")
            
            # Save analysis report
            self.db.save_analysis_report(
                os.path.basename(self.selected_audio_file),
                self.current_user['full_name'],
                best_match if best_match else {},
                best_confidence
            )
            
            self.results_text.insert(tk.END, f"\nAnalysis completed and saved at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            
        except Exception as e:
            messagebox.showerror("Analysis Error", f"Error analyzing audio file:\n{str(e)}")
            self.results_text.insert(tk.END, f"Error: {str(e)}\n")
    
    def refresh_frequency_database(self):
        """Refresh the frequency database view"""
        # Clear existing items
        for item in self.db_tree.get_children():
            self.db_tree.delete(item)
        
        # Get all frequency records
        records = self.db.get_all_frequency_records()
        
        for record in records:
            student_id, student_name, frequencies, session_date, duration = record
            freq_list = json.loads(frequencies)
            freq_str = ', '.join([f"{f:.1f}Hz" for f in freq_list])
            
            # Format duration
            if duration:
                duration_str = f"{duration//60}m {duration%60}s"
            else:
                duration_str = "N/A"
            
            # Format date
            try:
                formatted_date = datetime.fromisoformat(session_date).strftime('%Y-%m-%d %H:%M')
            except:
                formatted_date = session_date
            
            self.db_tree.insert('', 'end', values=(
                student_id, student_name, freq_str, formatted_date, duration_str
            ))
    
    def start_enhancement(self):
        if self.session_active:
            return
        
        # Generate frequencies
        frequencies = self.audio_enhancer.generate_frequencies(
            self.current_user['full_name'], 
            self.current_user['student_id']
        )
        
        # Start audio enhancement
        if self.audio_enhancer.start_enhancement(frequencies):
            # Add to database with proper cleanup first
            conn = sqlite3.connect(self.db.db_file)
            cursor = conn.cursor()
            
            # Remove any existing sessions for this student
            cursor.execute('DELETE FROM active_sessions WHERE student_id = ?', 
                          (self.current_user['student_id'],))
            
            # Add new session
            cursor.execute('''
                INSERT INTO active_sessions (student_id, student_name, frequencies, status)
                VALUES (?, ?, ?, ?)
            ''', (self.current_user['student_id'], self.current_user['full_name'], 
                  json.dumps(frequencies), 'active'))
            conn.commit()
            conn.close()
            
            self.session_active = True
            self.session_start_time = datetime.now()
            self.status_label.config(text="Status: Enhancement Active", fg='#27ae60')
            self.start_btn.config(state='disabled')
            self.stop_btn.config(state='normal')
            
            # Start ping thread
            self.ping_thread = threading.Thread(target=self.ping_session, daemon=True)
            self.ping_thread.start()
            
            messagebox.showinfo("Success", f"Audio enhancement activated!\nFrequencies: {', '.join([f'{f:.1f}Hz' for f in frequencies])}")
        else:
            messagebox.showerror("Error", "Failed to start audio enhancement")
    
    def stop_enhancement(self):
        if not self.session_active:
            return
        
        # Calculate session duration
        session_duration = 0
        if self.session_start_time:
            session_duration = int((datetime.now() - self.session_start_time).total_seconds())
        
        # Get frequencies for database storage
        frequencies = self.audio_enhancer.frequencies
        
        self.audio_enhancer.stop_enhancement()
        
        # Update database
        conn = sqlite3.connect(self.db.db_file)
        cursor = conn.cursor()
        cursor.execute('UPDATE active_sessions SET status = ? WHERE student_id = ?', 
                      ('ended', self.current_user['student_id']))
        conn.commit()
        conn.close()
        
        # Save to frequency database
        self.db.save_frequency_record(
            self.current_user['student_id'],
            self.current_user['full_name'],
            frequencies,
            session_duration
        )
        
        self.session_active = False
        self.status_label.config(text="Status: Ready", fg='#f39c12')
        self.start_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        
        messagebox.showinfo("Success", f"Audio enhancement stopped\nSession duration: {session_duration//60}m {session_duration%60}s")
    
    def ping_session(self):
        while self.session_active:
            conn = sqlite3.connect(self.db.db_file)
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE active_sessions SET last_ping = CURRENT_TIMESTAMP 
                WHERE student_id = ? AND status = 'active'
            ''', (self.current_user['student_id'],))
            conn.commit()
            conn.close()
            time.sleep(30)  # Ping every 30 seconds
    
    def refresh_sessions(self):
        if not hasattr(self, 'sessions_tree'):
            return
        
        # Clear existing items
        for item in self.sessions_tree.get_children():
            self.sessions_tree.delete(item)
        
        # Get active sessions
        sessions = self.db.get_active_sessions()
        
        # Update session count
        if hasattr(self, 'session_count_label'):
            self.session_count_label.config(text=f"Active Sessions: {len(sessions)}")
        
        for session in sessions:
            freq_str = ', '.join([f"{f:.1f}Hz" for f in session['frequencies']])
            duration = f"{session['duration']//60}m {session['duration']%60}s"
            
            # Format last ping time
            try:
                ping_time = datetime.fromisoformat(session['last_ping']).strftime('%H:%M:%S')
            except:
                ping_time = 'Unknown'
            
            self.sessions_tree.insert('', 'end', values=(
                session['student_id'],
                session['student_name'],
                freq_str,
                duration,
                ping_time
            ))
    
    def toggle_auto_refresh(self):
        if hasattr(self, 'auto_refresh_active'):
            self.auto_refresh_active = not self.auto_refresh_active
            if self.auto_refresh_active:
                self.auto_refresh_loop()
    
    def auto_refresh_loop(self):
        if hasattr(self, 'auto_refresh_active') and self.auto_refresh_active:
            self.refresh_sessions()
            # Schedule next refresh in 3 seconds
            self.root.after(3000, self.auto_refresh_loop)
    
    def logout(self):
        # Stop auto-refresh if active
        if hasattr(self, 'auto_refresh_active'):
            self.auto_refresh_active = False
            
        if self.session_active:
            self.stop_enhancement()
        
        self.current_user = None
        self.create_login_interface()
    
    def on_closing(self):
        if self.session_active:
            self.stop_enhancement()
        self.audio_enhancer.cleanup()
        self.root.destroy()

def main():
    root = tk.Tk()
    app = EchoShieldApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()