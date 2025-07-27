```
```
##   Multimodal Secure Transaction System
This project simulates a **secure transaction pipeline** using **multimodal machine learning**. It integrates **face recognition**, **voice authentication**, and **product recommendation** in one command-line interface, designed for interactive identity verification and recommendation.

---

---

##  Overview

A secure transaction involves three steps:

1. **Face Recognition** → Confirm user identity
2. **Voice Authentication** → Detect approval or denial
3. **Product Recommendation** → Suggest a suitable product

Each model was trained independently and integrated into a single decision pipeline.

---

##  Folder Structure

```

├── data/
│   ├── images/                 
│   ├── audio/                  
│   ├── image\_features.csv
│   ├── audio\_features.csv
│   └── merged\_dataset.csv
├── models/
│   ├── face\_model.py
│   ├── voice\_model.py
│   └── product\_model.py
├── notebooks/
│   └── pipeline\_notebook.ipynb
├── system\_demo.py             
├── requirements.txt
└── README.md

````

---

##  Dataset Summary

| Feature Set        | File Name           | Description                               |
|--------------------|---------------------|-------------------------------------------|
| Face Data          | `image_features.csv`| MFCC-like features extracted from images  |
| Voice Data         | `audio_features.csv`| MFCC, rolloff, and energy features        |
| Combined Dataset   | `merged_dataset.csv`| Merged facial + voice features + target   |

---

## Model Performance

###  Face Recognition Model (Random Forest)
| Metric     | Value   |
|------------|---------|
| Accuracy   | 0.8667  |
| F1 Score   | 0.8578  |
| Log Loss   | 0.2693  |

###  Voice Authentication Model (Random Forest)
| Metric     | Value   |
|------------|---------|
| Accuracy   | 0.7500  |
| F1 Score   | 0.7333  |
| Log Loss   | 0.5632  |

###  Product Recommendation Model (Random Forest)
| Metric     | Value   |
|------------|---------|
| Accuracy   | 0.2300  |
| F1 Score   | 0.2100  |
| Log Loss   | 1.8403  |

**Note**: The product recommendation model underperformed due to limited and imbalanced data (4 users × 3 product classes). Further data collection and balancing would improve results.

---

##  Running the System

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
````

### Step 2: Simulate a Transaction

```bash
python system_demo.py <name> <expression> <audio_file.wav>
# Example:
python system_demo.py Nelly smiling ./audio/Nelly/approve.wav
```

---


##  Demo Video

Watch the system in action (face → voice → product):
 [Demo Simulation Link](https://your-demo-link.com) 

---

##  Tools & Libraries

* Python, NumPy, Pandas
* `scikit-learn` for model training
* `librosa` for audio feature extraction
* `cv2`, `matplotlib`, `seaborn`
* Jupyter Notebook for model prototyping

---

##  Future Improvements

* Collect more user data and add diverse expressions
* Include speaker diarization and real-time webcam/audio input
* Train deep learning models (CNN, RNN) for improved accuracy

---


