import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import requests # Import requests

# Page configuration
st.set_page_config(
    page_title="Cybersecurity Threat Detection",
    page_icon="🔒",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
    <style>
        html, body {
            background-image: url('https://raw.githubusercontent.com/Praveen7-C/Epidemic-Cybersecurity-Threat-Modeling-with-ML/main/download.jpg') !important;
            background-size: cover;
            background-position: center;
            font-family: 'Arial', sans-serif;
            color: #34495E;
        }
        .title {
            font-size: 32px;
            font-weight: 700;
            color: #0000FF;
            text-align: center;
            padding: 40px 40px;
            background-color: rgba(236, 240, 241, 0.8);
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .section-header {
            font-size: 32px;
            color: #2980B9;
            font-weight: 600;
            margin-bottom: 20px;
            padding-bottom: 5px;
            border-bottom: 2px solid #2980B9;
        }
        .prediction-output {
            font-size: 26px;
            font-weight: bold;
            color: #D35400;
            text-align: center;
            padding: 20px;
            border: 2px solid #E74C3C;
            border-radius: 10px;
            background-color: #FADBD8;
        }
        .upload-container {
            background-color: #FFFFFF;
            padding: 30px;
            border-radius: 10px;
            margin: 20px 0;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .button-style {
            background-color: #2980B9;
            color: white;
            border-radius: 8px;
            padding: 12px 20px;
            font-size: 18px;
            font-weight: 600;
        }
        .button-style:hover {
            background-color: #3498DB;
        }
        .card {
            background-color: #ffffff;
            border-radius: 12px;
            padding: 20px;
            margin-top: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .card-header {
            font-size: 22px;
            font-weight: bold;
            color: #2980B9;
            margin-bottom: 10px;
        }
        .card-body {
            font-size: 16px;
            color: #7F8C8D;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        .input-label {
            font-size: 18px;
            font-weight: 500;
            color: #2C3E50;
        }
        .sidebar .sidebar-content {
            background-color: #34495E;
        }
        .sidebar .sidebar-header {
            color: #ECF0F1;
            font-size: 18px;
        }
    </style>
""", unsafe_allow_html=True)

# Function to fetch raw data from GitHub
@st.cache_data
def get_csv_data(url):
    response = requests.get(url)
    response.raise_for_status() # Raise an exception for bad status codes
    return response.content

# Streamlit app
def main():
    st.markdown("<div class='title'>Defense Strategies For Epidemic Cyber Security Threats Modeling And Analysis Using A Machine Learning Approach</div>", unsafe_allow_html=True)

    

    # Add download buttons in the sidebar
    st.sidebar.header("Download Sample Datasets")

    book_csv_url_raw = "https://raw.githubusercontent.com/Praveen7-C/Epidemic-Cybersecurity-Threat-Modeling-with-ML/main/Book.csv"
    dataset_csv_url_raw = "https://raw.githubusercontent.com/Praveen7-C/Epidemic-Cybersecurity-Threat-Modeling-with-ML/main/Dataset.csv"

    try:
        book_data = get_csv_data(book_csv_url_raw)
        st.sidebar.download_button(
            label="Download Book.csv",
            data=book_data,
            file_name="Book.csv",
            mime="text/csv"
        )
    except requests.exceptions.RequestException as e:
        st.sidebar.error(f"Could not download Book.csv: {e}")

    try:
        dataset_data = get_csv_data(dataset_csv_url_raw)
        st.sidebar.download_button(
            label="Download Dataset.csv",
            data=dataset_data,
            file_name="Dataset.csv",
            mime="text/csv"
        )
    except requests.exceptions.RequestException as e:
        st.sidebar.error(f"Could not download Dataset.csv: {e}")

    st.sidebar.markdown("---")
    st.sidebar.header("Upload Your Dataset")
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")


    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.subheader("Dataset Overview")
        st.write(df.head())

        if st.button("Preprocess Data", key="preprocess"):
            st.subheader("Preprocessing Data")
            df = df.dropna()

            if 'label_encoder_proto' not in st.session_state:
                st.session_state.label_encoder_proto = LabelEncoder()
            if 'label_encoder_state' not in st.session_state:
                st.session_state.label_encoder_state = LabelEncoder()
            if 'label_encoder_attack_cat' not in st.session_state:
                st.session_state.label_encoder_attack_cat = LabelEncoder()

            df['proto'] = st.session_state.label_encoder_proto.fit_transform(df['proto'])
            df['state'] = st.session_state.label_encoder_state.fit_transform(df['state'])
            df['attack_cat'] = st.session_state.label_encoder_attack_cat.fit_transform(df['attack_cat'])

            st.write("Data Preprocessing Completed.")
            st.write(df.head())

            X = df.drop(columns=['id', 'label'])
            y = df['label']
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            st.session_state.X_scaled = X_scaled
            st.session_state.y = y
            st.session_state.scaler = scaler

            st.write("Preprocessed data is ready.")

        if 'X_scaled' in st.session_state and 'y' in st.session_state:
            st.sidebar.markdown("---")
            n_estimators = st.sidebar.number_input("Number of Estimators", 10, 200, 100)
            max_depth = st.sidebar.number_input("Max Depth", 1, 50, 10)

            if st.button("Train Model", key="train_model"):
                X_train, X_test, y_train, y_test = train_test_split(
                    st.session_state.X_scaled, st.session_state.y, test_size=0.2, random_state=42
                )
                model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
                model.fit(X_train, y_train)

                st.session_state.model = model
                st.session_state.X_test = X_test
                st.session_state.y_test = y_test

                st.write("Model training completed.")

        if 'model' in st.session_state:
            if st.button("Evaluate Model", key="evaluate_model"):
                model = st.session_state.model
                y_pred = model.predict(st.session_state.X_test)

                accuracy = accuracy_score(st.session_state.y_test, y_pred)
                st.write(f"Accuracy: {accuracy*100:.2f}%")

                st.subheader("Confusion Matrix")
                cm = confusion_matrix(st.session_state.y_test, y_pred)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(st.session_state.y_test), yticklabels=np.unique(st.session_state.y_test))
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                st.pyplot(fig)

                st.subheader("Classification Report")
                st.text(classification_report(st.session_state.y_test, y_pred))

                st.subheader("Feature Importance")
                feature_data = pd.DataFrame({
                    'Feature': df.drop(columns=['id', 'label']).columns,
                    'Importance': model.feature_importances_
                }).sort_values(by='Importance', ascending=False)

                st.write(feature_data)
                st.bar_chart(feature_data.set_index('Feature')['Importance'])

        if 'model' in st.session_state:
            st.subheader("Enter Input Values for Prediction")

            dur = st.number_input("Duration (dur)", 0, 1000, 10)
            proto = st.selectbox("Protocol (proto)", options=df['proto'].unique())
            state = st.selectbox("State (state)", options=df['state'].unique())
            spkts = st.number_input("Source Packets (spkts)", 0, 10000, 100)
            dpkts = st.number_input("Destination Packets (dpkts)", 0, 10000, 100)
            sbytes = st.number_input("Source Bytes (sbytes)", 0, 1000000, 500)
            dbytes = st.number_input("Destination Bytes (dbytes)", 0, 1000000, 500)
            rate = st.number_input("Rate (rate)", 0.0, 1000000.0, 1.0)
            attack_cat = st.selectbox("Attack Category (attack_cat)", options=df['attack_cat'].unique())

            if st.button("Predict Label", key="predict"):
                proto_encoded = st.session_state.label_encoder_proto.transform([proto])[0]
                state_encoded = st.session_state.label_encoder_state.transform([state])[0]
                attack_cat_encoded = st.session_state.label_encoder_attack_cat.transform([attack_cat])[0]

                input_data = np.array([[dur, proto_encoded, state_encoded, spkts, dpkts, sbytes, dbytes, rate, attack_cat_encoded]])
                input_data_scaled = st.session_state.scaler.transform(input_data)

                predicted_label = st.session_state.model.predict(input_data_scaled)

                if predicted_label[0] == 0:
                    st.markdown("<div class='prediction-output'>No Cyber Threat Detected</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div class='prediction-output'>Cyber Threat Detected</div>", unsafe_allow_html=True)

if __name__ == '__main__':
    main()
