#NLP-Based Depression Detection System (DistilBERT Fine-Tuned Model)

This project focuses on detecting **signs of depression through text analysis** using advanced **Natural Language Processing (NLP)** techniques.  
It fine-tunes the **DistilBERT transformer model** to classify text as either:

- **Depressed**
- **Not Depressed**

The model is deployed through a **Streamlit web application**, allowing real-time user input and prediction.

---

## 🚀 Features
- Uses **DistilBERT**, a lightweight transformer optimized for performance.
- Detects emotional and psychological distress in text inputs.
- Clean, interactive **Web UI using Streamlit**.
- Shows confidence score for predictions.
- Trained on a processed mental-health dataset.

---

## 🧰 Tech Stack
| Component | Technology Used |
|----------|----------------|
| Language | Python |
| NLP Model | DistilBERT (HuggingFace Transformers) |
| Dataset Handling | Pandas, HuggingFace Datasets |
| Training Framework | PyTorch + HuggingFace Trainer |
| Web App UI | Streamlit |
| Evaluation | Accuracy, Precision, Recall, F1-Score |

---

## 📂 Project Structure

nlp_mental_health/

│

├── data/

│ ├── raw/ # Original dataset

│ └── processed/ # Cleaned + Split (train/test)

├── models/

│ └── distilbert_model/ # Saved fine-tuned model + tokenizer

├── src/

│ ├── prepare_data.py # Cleans + splits dataset

│ └── train_distilbert.py # Full model training script


├── webapp/

│ └── app.py # Streamlit UI


└── README.md

🤝 Contribution
Contributions, issues and feature requests are welcome!
Feel free to fork the repo & submit PRs.

⭐ Show Support
If you found this project useful, give it a star ⭐ on GitHub!

🧡 Disclaimer
This project is built for educational and research purposes only.
It cannot replace professional mental health diagnosis or therapy
