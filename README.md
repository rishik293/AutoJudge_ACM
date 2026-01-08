# AutoJudge – Problem Difficulty Classification & Scoring System

## Project Overview
AutoJudge is a machine learning–based system that predicts the difficulty level
(Easy / Medium / Hard) and a numerical difficulty score (1–10) for competitive
programming problems based on their textual descriptions.

The system uses natural language processing techniques to analyze problem
statements and provides instant predictions through an interactive web interface.

---

## Dataset Used
- Custom dataset of programming problem statements
- Each sample contains:
  - Problem title
  - Description
  - Input format
  - Output format
- Labels:
  - Classification label: Easy / Medium / Hard
  - Regression label: Difficulty score (1–10)

---

## Approach and Models Used

### Data Preprocessing
- Text normalization (lowercasing, whitespace cleanup)
- Concatenation of title, description, input, and output
- Removal of irrelevant characters

### Feature Extraction
- TF-IDF Vectorization to convert text into numerical features

### Models
- **Classification Model**: Predicts problem difficulty category
- **Regression Model**: Predicts numerical difficulty score

Trained models are saved and reused during inference.

---

## Evaluation Metrics
- Classification Accuracy: *(mention your actual value here)*  
- Mean Absolute Error (MAE): *(value)*  
- Root Mean Squared Error (RMSE): *(value)*  

---

## Web Interface
- Built using **Streamlit**
- Allows users to paste a coding problem description
- Predicts:
  - Difficulty category (Easy / Medium / Hard)
  - Difficulty score (1–10)
- Results are displayed instantly on the web UI

---

## Steps to Run the Project Locally

```bash
git clone <your-github-repo-link>
cd AUTOJUDGE
pip install -r requirements.txt
streamlit run app.py

Demo Video:
📹 Demo Video Link: [Paste your 2–3 min video link here]

##My Details
Name: Your Name
Institute: IIT Roorkee
Project Type: Machine Learning & NLP