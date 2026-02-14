# AI-Based Frustration Detection System for Developers

An AI-powered system that detects developer frustration during coding sessions using **keystroke dynamics** and **facial emotion recognition**, with a gamified support mechanism and team health dashboard.

---

## Problem Statement

Developers often experience stress and frustration during debugging and code reviews, which impacts productivity and mental well-being.

This project aims to detect frustration in real-time using behavioral signals and provide supportive interventions.

---

## Solution Overview

The system:

- Collects keystroke dynamics data
- Analyzes facial expressions
- Uses LightGBM for frustration classification
- Displays individual & team emotional health dashboard
- Includes a gamified assistance layer

---

## Project Phases

1. **Data Collection**
   - Keystroke logging
   - Structured session-based labeling

2. **Feature Engineering**
   - Typing speed
   - Backspace frequency
   - Pause detection
   - Burst typing patterns

3. **Model Training**
   - LightGBM classifier
   - Performance evaluation
   - Feature importance analysis

4. **Facial Emotion Detection**
   - OpenCV / MediaPipe integration
   - Emotion probability scoring

5. **Backend System**
   - Session storage
   - Frustration scoring API

6. **Frontend Dashboard**
   - Daily/weekly stress trends
   - Team health index visualization

7. **Gamification Layer**
   - Break reminders
   - Mini productivity tasks
   - Reward system

---

## Tech Stack

- **Python**
- LightGBM
- OpenCV / MediaPipe
- Flask / Django (Backend)
- MySQL
- JavaScript (IDE Extension)

---

## Dataset Structure (Example)

| typing_speed | backspace_count | pause_count | avg_pause | burst_ratio | label |
|--------------|-----------------|------------|-----------|-------------|-------|

Label:
- 0 → Calm
- 1 → Frustrated

---

##  Ethical Considerations

- No actual code content is stored.
- Only behavioral patterns are analyzed.
- Data is anonymized.
- User consent required.

---
##  Future Enhancements

- Multi-user training dataset
- Deep learning emotion fusion
- Real-time IDE plugin deployment
- Enterprise-level analytics

---


--
