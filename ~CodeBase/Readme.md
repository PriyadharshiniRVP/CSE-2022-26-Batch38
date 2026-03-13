# AI-Based Frustration Detection System for Developers

An AI-powered system that detects developer frustration during coding sessions using **keystroke dynamics** and **facial emotion recognition**, combined with a **gamified support mechanism** and **team health dashboard**.

---

## Problem Statement

Developers often experience stress and frustration during debugging and code reviews, which impacts productivity and mental well-being.

This project aims to:
- Detect frustration in real-time using behavioral signals
- Provide supportive interventions
- Promote healthier coding environments

---

## Solution Overview

The system:

- Collects keystroke dynamics data
- Analyzes facial expressions
- Uses LightGBM for frustration classification
- Displays individual & team emotional health dashboard
- Includes a gamified assistance layer

---

## 🧩 Project Phases

###  Data Collection
- Keystroke logging
- Structured session-based labeling

### Feature Engineering
- Typing speed
- Backspace frequency
- Pause detection
- Burst typing patterns

### Model Training
- LightGBM classifier
- Performance evaluation
- Feature importance analysis

### Facial Emotion Detection
- OpenCV / MediaPipe integration
- Emotion probability scoring

### Backend System
- Session storage
- Frustration scoring API

### Frontend Dashboard
- Daily / Weekly stress trends
- Team health index visualization

### Gamification Layer
- Break reminders
- Mini productivity tasks
- Reward system

---

## Tech Stack

### Backend
- Python
- LightGBM
- Flask / Django
- MySQL

### Computer Vision
- OpenCV
- MediaPipe

### Frontend / Integration
- JavaScript (IDE Extension)

---

## Dataset Structure (Example)

| typing_speed | backspace_count | pause_count | avg_pause | burst_ratio | label |
|--------------|----------------|------------|-----------|-------------|-------|

**Label Encoding:**
- `0` → Calm  
- `1` → Frustrated  

---

## Ethical Considerations

- No actual code content is stored
- Only behavioral patterns are analyzed
- Data is anonymized
- User consent is required

---

## Future Enhancements

- Multi-user training dataset
- Deep learning–based emotion fusion
- Real-time IDE plugin deployment
- Enterprise-level analytics

---

## Vision

To build emotionally intelligent development environments that support productivity, well-being, and team collaboration.

---

Built to make coding smarter, healthier, and more human.