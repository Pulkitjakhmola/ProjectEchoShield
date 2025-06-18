---

# 🎙 Echo Shield

> **Anti-Recording Defense for Lectures**
> A system to protect in-person and online lectures from unauthorized recordings by injecting subtle audio distortions that are inaudible to humans but degrade recordings.

---

##  Project Overview

**Echo Shield** introduces a novel method to secure lectures from illegal sharing or recording. It works by generating high-frequency noise or phase-shifted echoes using Python, which interfere with recording devices without affecting human listeners.

###  Key Features

*  Inaudible high-frequency audio distortions
*  Works both in-person (via speakers) and online (e.g., Zoom audio stream)
*  Includes unique **distortion fingerprints** for traceability
*  Tested with phone recorders to ensure distortion effectiveness

---

##  Requirements

Install dependencies via `pip` using the provided `requirements.txt`:

```bash
pip install -r requirements.txt
```

###  `requirements.txt`

```
flask==3.0.3
numpy==1.26.4
librosa==0.10.2
soundfile==0.12.1
scipy==1.13.1
matplotlib==3.9.0
werkzeug==3.0.3
```

---

##  Getting Started

Once dependencies are installed, simply run:

```bash
python EchoShield.py
```

The Flask server will start and be accessible at:

```
http://localhost:5000/
```

---

##  How It Works

* **Distortion Generation**: Uses `librosa` and `PyDub` to synthesize high-frequency noise and subtle echo shifts.
* **Delivery**:

  * In-person: Played through speakers
  * Online: Streamed via conferencing tools (e.g., Zoom)
* **Fingerprinting**: Each lecture embeds a unique audio watermark for forensic tracing.

---

##  Project Structure 
```
EchoShield/
├── EchoShield.py
├── requirements.txt
├── static/
│   └── audio_samples/
├── templates/
│   └── index.html
└── README.md
```

---

##  Contributing

We welcome contributions! Feel free to:

* Open issues or feature requests
* Submit pull requests for enhancements or bug fixes

---

##  License

MIT License 

---
