# 🤖 Intelligent Assistance with Face Recognition

> Final Year B.Tech Project | Gandhinagar Institute of Technology
> Created by **Dhruv Panchal** during internship at **Kody Technolab Pvt. Ltd.**

---

## 📌 Project Overview

This project is a smart, interactive assistant designed to operate within a **college or academic environment**, combining **face recognition**, **voice-based communication**, and a **college-specific chatbot**.

The system:

* Detects and identifies people via **face recognition** (OpenCV)
* Greets the person by name if recognized, or generically if unknown
* Converts **spoken queries into text**
* Responds using **natural language processing** trained on a **college-specific dataset**

---

## 🧠 Key Features

| Module                               | Description                                                   |
| ------------------------------------ | ------------------------------------------------------------- |
| 🎥 **Face Detection & Recognition** | Detects and identifies individuals from camera feed           |
| 🗣️ **Speech-to-Text**               | Converts voice queries into text for processing               |
| 💬 **NLP Chatbot**                  | Understands user queries and responds using trained model     |
| 📚 **College Knowledgebase**        | Trained on custom dataset covering 50+ college-related topics |
| 🔊 **Text-to-Speech**               | Responds back verbally using TTS engine                       |
| 💡 **Context-Free**                 | Doesn’t require prior instruction tuning                      |

---

## 💠 Technologies Used

* **Python 3**
* **OpenCV** – Face detection and recognition
* **NLTK / spaCy** – NLP preprocessing
* **Keras + TensorFlow** – Custom intent classification model
* **SpeechRecognition / PyAudio** – Voice input
* **pyttsx3 / gTTS** – Voice output
* **Matplotlib** – Training graphs
* **Pickle & JSON** – Data serialization
* **Jupyter / VSCode** – Development environments

---

## 🗂️ Project Structure

```
IntelligentAssistant/
├── training_data/
│   └── data.json              # Dataset with 52 tags, 160+ patterns
├── model_data/
│   ├── model.keras            # Trained NLP model
│   ├── words.pkl              # Tokenized vocabulary
│   └── classes.pkl            # Output tags (intents)
├── train/
│   └── training.py            # Code to train intent classifier
├── inference/
│   └── prediction.py          # Code to predict tag from user input
├── main.py                    # (To be added) Unified pipeline: face → speech → NLP → TTS
├── README.md                  # Project description and instructions
```

---

## 🦪 Dataset Snapshot

* **Total Tags (Intents)**: 52
* **Total Patterns**: 163
* **Sample Tags**:

  * `greeting`, `identity`, `goodbye`
  * `courses`, `fees`, `admission`
  * `library`, `placement`, `facilities`
  * `weekend_activities`, `favorite_class`, `internship_opportunities`, etc.

> Example Pattern: *“What are the courses offered?”*
> Response: *“The college offers IT, CE, ME, CH, and Civil Engineering...”*

---

## 🚀 How to Run

### 1️⃣ Train the Chatbot Model

```bash
cd train
python training.py
```

This generates:

* `model.keras`
* `words.pkl`
* `classes.pkl`
* `model_loss_graph.png`
* `model_accuracy_graph.png`

---

### 2️⃣ Predict Responses (Text-Based)

```bash
cd inference
python prediction.py
```

Use `chatbot_response("your message")` to test.

---

### 3️⃣ (Optional) Create Main Application

Integrate:

* Webcam feed
* Voice input/output
* NLP engine

---

## 🧠 Sample Interaction

```
User: (Camera detects face) → Recognized as "Dhruv"
Bot: Hello Dhruv! How can I help you today?

User: (Speaks) What are the college timings?
Bot: The college is open from 9am to 3pm, Monday to Saturday.
```

---

## ✅ Achievements

* ✅ Reached \~90% face recognition accuracy
* ✅ Reached \~97% chatbot classification accuracy
* ✅ Voice-based system working end-to-end
* ✅ Custom domain-specific dataset for college environment

---

## 📌 Future Work

* Switch NLP model to **Transformers (BERT/Gemma/DistilBERT)**
* Add **memory or context handling**
* Build **GUI/Web interface** using Streamlit or Flask
* Enhance dataset with real-world phrasing and queries
* Deploy on a Raspberry Pi + Camera for on-campus assistant

---

## License

This project is developed as part of academic work and is open for educational use.


