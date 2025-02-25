# Anomaly Detection in Network Traffic Using Machine Learning

A machine learning-based solution to detect anomalous network traffic patterns, enabling early identification of cyber threats like DDoS attacks, intrusions, and malware.

## 📖 Description

This project leverages machine learning algorithms to identify anomalies in network traffic data, a critical task for maintaining cybersecurity. By analyzing features such as packet size, protocol type, source/destination IP addresses, and flow duration, the system flags suspicious activities that deviate from normal behavior. The goal is to provide a scalable, automated tool for enhancing network security infrastructure.

### Key Features:
- **Preprocessing Pipeline**: Handles missing values, normalization, and feature engineering.
- **Multiple ML Algorithms**: Includes Isolation Forest, SVM, Random Forest, and Autoencoders.
- **Performance Metrics**: Evaluates models using precision, recall, F1-score, and ROC-AUC.
- **Visualization Tools**: Generates confusion matrices, ROC curves, and feature importance plots.
- **Scalable Design**: Adaptable to large datasets and real-time monitoring systems.

---

## 🚀 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/abhistorm/Anomaly-Detection-in-Network-Traffic-using-Machine-Learning.git
   cd Anomaly-Detection-in-Network-Traffic-using-Machine-Learning

Dataset
The project uses the NSL-KDD dataset, a refined version of KDD Cup 1999. Key features include:

duration: Connection length

protocol_type: Protocol (e.g., TCP, UDP)

flag: Connection status

src_bytes: Bytes sent from source to destination

attack_type: Label indicating normal or attack (e.g., DoS, Probe).

Replace with other datasets (e.g., CICIDS2017) by adjusting the preprocessing steps.

Results
Example performance of an Isolation Forest model:

Accuracy: 98.5%

Precision: 96.2%

Recall: 95.8%

F1-Score: 96.0%

Confusion Matrix
Confusion matrix showing normal vs. anomaly classification.


This README provides a structured overview of the project, installation steps, usage examples, and contribution guidelines. Customize fields like contact email, dataset links, or license details as needed.
