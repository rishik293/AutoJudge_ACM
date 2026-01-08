import streamlit as st
import joblib

# 1. Load the trained models
try:
    classifier = joblib.load('model_class.pkl')
    regressor = joblib.load('model_score.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
except:
    st.error("Error: Models not found. Please run 'train.py' first.")
    st.stop()

# 2. Configure the Page
st.set_page_config(page_title="AutoJudge AI", page_icon="🧠")

st.title("🧠 AutoJudge: Difficulty Predictor")
st.markdown("Paste a coding problem below to see if it's **Easy**, **Medium**, or **Hard**.")

# 3. Create Input Boxes
col1, col2 = st.columns(2)

with col1:
    title = st.text_input("Problem Title (Optional)")
    desc = st.text_area("Problem Description", height=200, placeholder="e.g. Write a program to sort an array...")

with col2:
    inp_desc = st.text_area("Input Description", height=80, placeholder="e.g. First line contains integer N...")
    out_desc = st.text_area("Output Description", height=80, placeholder="e.g. Print the sorted array...")

# 4. Predict Button
if st.button("Predict Difficulty", type="primary"):
    if not desc:
        st.warning("Please enter a description first!")
    else:
        # Combine the text exactly how we did in training
        # We join Title + Description + Input + Output
        full_text = f"{title} {desc} {inp_desc} {out_desc}"
        
        # Convert text to numbers
        text_vectorized = vectorizer.transform([full_text])
        
        # Make Predictions
        pred_class = classifier.predict(text_vectorized)[0]
        pred_score = regressor.predict(text_vectorized)[0]
        
        # 5. Show Results nicely
        st.divider()
        st.subheader("Prediction Results")
        
        # Color code the result
        if pred_class == "Easy":
            color = "green"
        elif pred_class == "Medium":
            color = "orange"
        else:
            color = "red"
            
        st.markdown(f"### Predicted Class: :{color}[{pred_class}]")
        
        # Display the RAW score (e.g., 2.4 or 8.1)
        st.metric(label="Difficulty Score (1-10 Scale)", value=f"{pred_score:.2f}")