# Epidemic-Cybersecurity-Threat-Modeling-with-ML

## Project Overview

This project integrates **epidemic modeling concepts** with **cybersecurity threat analysis** using **Machine Learning** to predict and understand how cyber threats spread across computer networks.  
By treating cyber-attacks like infectious diseases, the system models threat propagation, identifies vulnerable systems, and supports proactive security decision-making.


**Live Deployment (Streamlit App)**  
https://epidemic-cybersecurity-threat-modeling-with-ml.streamlit.app/

---

## Features

- **Cyber Threat Modeling** using Machine Learning
- **Epidemic-Based Simulation** of threat propagation across networks
- **Data Analysis** on real-world cybersecurity datasets
- **Predictive Analytics** for forecasting future threat behavior
- **Interactive Web Dashboard** for visualization and insights
- **Multi-Dataset Support** for training and validation

---

## Datasets

The project uses the following datasets:

- **Dataset.csv**  
  A comprehensive cybersecurity threat dataset containing attack types, attributes, and classifications.

- **Book.csv**  
  A reference dataset used for validation and comparative threat modeling.

These datasets include information related to attack vectors, vulnerabilities, and system behaviors.

---

## Installation

### Prerequisites

- Python 3.12
- pip (Python package manager)
- Virtual environment (recommended)

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/Praveen7-C/Epidemic-Cybersecurity-Threat-Modeling-with-ML.git
   cd Epidemic-Cybersecurity-Threat-Modeling-with-ML
   ```

2. **Create and activate a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate      # On Linux/Mac
   venv\Scripts\activate         # On Windows
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

Run the application using:

```bash
python app.py
```

Once started, the web application will be accessible at:

```
http://localhost:5000
```

---

## Project Structure

```
Epidemic-Cybersecurity-Threat-Modeling-with-ML/
│
├── app.py              # Main application file
├── Dataset.csv         # Primary cybersecurity dataset
├── Book.csv            # Reference dataset
├── download.jpg        # Visualization / architecture diagram
├── requirements.txt
└── README.md           # Project documentation
```

---

## Machine Learning Concepts Used

* Threat Classification
* Anomaly Detection
* Predictive Modeling
* Network-Based Threat Propagation
* Epidemic Models (SIR / SEIR inspired approaches)

---

## Key Concepts

* **Epidemic Modeling**: Cyber threats are modeled like infectious diseases.
* **Threat Propagation**: Simulates how malware or attacks spread across systems.
* **Risk Assessment**: Identifies high-risk nodes and vulnerabilities.
* **Decision Support**: Helps security teams plan mitigation strategies.

---

## Technologies Used

* **Python**
* **Machine Learning**: scikit-learn
* **Data Processing**: Pandas, NumPy
* **Visualization**: Matplotlib, Seaborn
* **Web Framework**: Streamlit / Flask

---

## Use Cases

* Cybersecurity risk assessment
* Network vulnerability analysis
* Threat spread simulation
* Academic research and learning
* Security awareness and training

---

## Contributing

Contributions are welcome!

1. Fork the repository
2. Create a new branch (`feature/your-feature-name`)
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

---

## License

This project is licensed under the **MIT License**.

---

For any questions or issues, feel free to contact [gmail](nagaraju736881@gmail.com).
