---

# User Identity and Product Recommendation System

## Overview

This project simulates a secure product recommendation pipeline using **multimodal authentication** — combining facial recognition and voice verification. Only authenticated users can access personalized product recommendations.

---

##  Features

*  **User Authentication** via:

  * Face recognition (neutral, smiling, surprised)
  * Voice verification using MFCC features
*  **Data Merging**:

  * Combines social profile and transaction data using standardized Customer ID
*  **Feature Extraction**:

  * Histograms for images (`image_features.csv`)
  * MFCCs for audio (`audio_features.csv`)
*  **Prediction Pipeline**:

  * Random Forest / Logistic Regression / XGBoost
* Access denied for failed authentication
* Access granted for verified users only

---

## Tech Stack

* **Python** (NumPy, pandas, OpenCV, librosa, scikit-learn, matplotlib)
* **Jupyter Notebook**
* **CSV** for feature storage

---

## Project Structure

```
├── Formative 2/
│   ├── images/
│   ├── audio/
│ ──data
     ├── social_profiles.csv
│    ├── transactions.csv

```

---

##  How to Clone

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

---

##  Steps to Reproduce

1. Load image and audio data
2. Extract image (histogram) and audio (MFCC) features
3. Merge datasets on standardized Customer ID
4. Train and evaluate prediction models
5. Simulate user logins with successful and failed cases
6. Output recommendation for verified users only

---

##  Notes

* Make sure to install dependencies listed in `requirements.txt`
* For testing, preloaded samples are included
* Update paths if adding new data

---
