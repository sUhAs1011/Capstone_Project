# 🎥 Detection and Mitigation of Replay Attacks in CCTV Systems

An AI-powered cybersecurity framework for detecting and mitigating replay attacks in CCTV surveillance systems using **Hierarchical Temporal Memory (HTM)** and **SHA-256 integrity verification**.

This project was developed as our **Final Year Capstone Project** at **PES University** and was subsequently published in the **IEEE Xplore Digital Library**. The proposed framework enhances surveillance security by identifying replay attacks in real time and automatically mitigating compromised video streams through intelligent anomaly detection and cryptographic verification.

---

# 📚 Research Publication

📄 **Published in IEEE Xplore**

**Paper Title:**  
*Detection and Mitigation of Replay Attacks in CCTV Systems*

🔗 **IEEE Xplore Paper:**  
https://ieeexplore.ieee.org/document/11511006

**Research Areas**

- Artificial Intelligence
- Cybersecurity
- Computer Vision
- Intelligent Surveillance Systems

---

# 🚀 Features

- 🎥 Real-time CCTV video stream monitoring
- 🧠 AI-powered replay attack detection using Hierarchical Temporal Memory (HTM)
- 🔐 SHA-256-based integrity verification
- 📹 Perceptual hashing for video authentication
- 🚨 Automated replay attack detection and mitigation
- 📊 High detection accuracy with low false positives
- ⚡ Modular and scalable architecture
- 🔒 Secure surveillance framework

---

# 🏗️ System Architecture

The proposed framework continuously analyzes incoming CCTV video streams, extracts temporal patterns using HTM, verifies video integrity, and detects replay attacks in real time.

## Workflow

```text
Video Stream
      │
      ▼
Hierarchical Temporal Memory (HTM)
      │
      ▼
Perceptual Hashing
      │
      ▼
SHA-256 Integrity Verification
      │
      ▼
Replay Attack Detection
      │
      ▼
Real-Time Mitigation
```

---

# 🧠 Methodology

## 1️⃣ Video Stream Acquisition

The system continuously receives live video streams from CCTV cameras.

---

## 2️⃣ Hierarchical Temporal Memory (HTM)

The incoming video stream is analyzed using **Hierarchical Temporal Memory (HTM)** to learn normal temporal patterns.

HTM enables the framework to detect anomalies by identifying unusual frame sequences that may indicate replay attacks.

---

## 3️⃣ Perceptual Hashing

A perceptual hash is generated for every video frame to capture its visual characteristics while remaining robust against minor image variations.

This enables efficient frame comparison during replay attack detection.

---

## 4️⃣ SHA-256 Integrity Verification

The generated perceptual hash is validated using **SHA-256** to ensure the authenticity and integrity of the transmitted video stream.

Any modification or replay attempt results in a mismatch during verification.

---

## 5️⃣ Replay Attack Detection & Mitigation

If anomalies and integrity mismatches are detected simultaneously, the system:

- Detects replay attacks in real time
- Alerts administrators
- Blocks compromised streams
- Initiates mitigation procedures

---

# 📊 Key Contributions

- Developed an AI-driven replay attack detection framework.
- Utilized **Hierarchical Temporal Memory (HTM)** for temporal anomaly detection.
- Combined perceptual hashing with **SHA-256** for secure stream verification.
- Enabled automated replay attack detection and mitigation.
- Designed a modular framework compatible with existing CCTV infrastructures.
- Improved surveillance reliability through intelligent cybersecurity techniques.

---

# 📈 Experimental Results

The proposed framework demonstrated:

- High replay attack detection accuracy
- Improved precision compared to traditional approaches
- Low false-positive rate
- Real-time attack detection capability
- Immediate mitigation after attack identification

The comparison study shows that the proposed approach achieves competitive performance while strengthening CCTV security using AI and cryptographic validation.

---

# 🛠️ Tech Stack

| Category | Technologies |
|----------|--------------|
| **Programming Language** | Python |
| **AI Framework** | Hierarchical Temporal Memory (HTM) |
| **Security** | SHA-256 |
| **Computer Vision** | Perceptual Hashing |
| **Application Domain** | CCTV Surveillance |
| **Research** | Artificial Intelligence & Cybersecurity |

---

# 📊 Research Poster

<p align="center">
  <img src="https://github.com/user-attachments/assets/3b4881e5-7d2d-4760-be00-23a89fd2446d" alt="Research Poster" width="900"/>
</p>

---

# 👥 Team Members

- **Mohit Prasad Singh** *(PES2UG22CS320)*
- **Shreyas Suresh** *(PES2UG22CS540)*
- **Soumya Ranjan Mishra** *(PES2UG22CS571)*
- **Suhas Venkata Karamalaputti** *(PES2UG22CS590)*

### Faculty Guide

- **Dr. Manju**
- PES University, Bengaluru

---

# 🎓 Academic Contribution

This project was developed as our **Bachelor of Technology Capstone Project** at **PES University** and contributes towards improving the security of intelligent surveillance systems through the integration of **Artificial Intelligence**, **Machine Learning**, and **Cryptographic Security**.

---

# 📄 Citation

If you find this work useful, please consider citing our paper published in **IEEE Xplore**.

📄 **Paper:** *Detection and Mitigation of Replay Attacks in CCTV Systems*

🔗 https://ieeexplore.ieee.org/document/11511006
