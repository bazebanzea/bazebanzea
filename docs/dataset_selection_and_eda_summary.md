# Dataset Selection and Exploratory Data Analysis (EDA) Summary

## Dataset Selection

### Chosen Dataset
**ToN_IoT (Telemetry of Network - Internet of Things)**

### Alternative Considered
*   **BoT-IoT**: Another popular dataset for IoT security, but ToN_IoT was chosen for its broader scope and more diverse data types.

### Reasoning for Selecting ToN_IoT
The ToN_IoT dataset was selected due to several key characteristics that make it suitable for developing a robust machine learning-based Intrusion Detection System (IDS) for IoT environments:

1.  **Heterogeneity**: It includes a wide variety of data sources, not just network traffic, but also telemetry data from IoT devices and system logs from operating systems. This provides a more holistic view of the IoT environment.
2.  **IoT/IIoT Focus**: The dataset is specifically generated from a realistic, large-scale IoT and Industrial IoT (IIoT) testbed, making it highly relevant to the project's domain.
3.  **Multiple Data Formats**: Data is available in several formats, including raw packet captures (pcaps), processed CSV files, and logs, offering flexibility for different analysis approaches.
4.  **Variety of Attack Scenarios**: It encompasses a comprehensive set of modern cyber-attacks relevant to IoT, such as various types of DoS/DDoS, scanning, man-in-the-middle, ransomware, and specific attacks against IoT protocols.
5.  **Labeled Data**: The dataset provides labeled instances, distinguishing between normal and malicious activities, which is essential for supervised machine learning.

## ToN_IoT Dataset Overview

### Origin
The ToN_IoT dataset was developed by the Cyber Range and IoT Labs at the UNSW Canberra Cyber, University of New South Wales (UNSW), Australia. It was generated from a sophisticated testbed environment designed to simulate a realistic, medium-scale IoT network, incorporating various IoT devices and services.

### Types of Data Included
The dataset is rich and diverse, containing:

*   **Network Traffic Data**: Full packet captures (PCAP files) and processed network flow data in CSV format (e.g., `Train_Test_Network.csv`). This includes metadata extracted from network communications like source/destination IPs, ports, protocols, and flow statistics.
*   **Telemetry Data**: Data collected from various IoT devices, reflecting their operational states and measurements. This data is often specific to the device type and its sensing capabilities.
*   **Operating System (OS) Logs**: Logs collected from Linux and Windows operating systems that were part of the testbed, offering insights into system-level activities and potential compromises.
*   **Security Information and Event Management (SIEM) Logs**: Aggregated logs that can provide a higher-level view of security events.

### Attack Categories
The ToN_IoT dataset includes a wide range of legitimate and illegitimate activities, with attack categories such as (but not limited to):

*   Denial of Service (DoS)
*   Distributed Denial of Service (DDoS)
*   Scanning (e.g., port scanning)
*   Man-in-the-Middle (MitM)
*   Ransomware
*   Cross-site Scripting (XSS)
*   Password cracking attacks
*   Injection attacks
*   Backdoor attacks
*   Specific IoT protocol attacks

### Availability
The dataset is typically made available through the UNSW Canberra Cyber website. It often includes:

*   Raw PCAP files.
*   Processed CSV files suitable for direct use with machine learning tools.
*   Pre-defined train/test splits to facilitate comparable research. (e.g., `Train_Test_Network.csv` which concatenates train and test sets with a 'label' or 'type' column to differentiate).

## Exploratory Data Analysis (EDA) Summary

### Reference Notebook
The initial exploratory data analysis for this project is conducted in the Jupyter notebook:
[`notebooks/01_initial_data_exploration.ipynb`](../notebooks/01_initial_data_exploration.ipynb)

*(Note: The link above is a relative path assuming this document is in `docs/` and the notebook in `notebooks/`)*

### EDA Steps Performed in the Notebook
The `01_initial_data_exploration.ipynb` notebook is structured to perform the following initial EDA steps once the `Train_Test_Network.csv` file from the ToN_IoT dataset is correctly placed in `data/raw/ton_iot/`:

1.  **Load Data**: The dataset is loaded into a pandas DataFrame. Error handling is included for cases where the file is not found.
2.  **Display Head**: The first few rows of the DataFrame are displayed (`df.head()`) to get a quick overview of the data structure and feature names.
3.  **DataFrame Info**: `df.info()` is used to get a concise summary of the DataFrame, including data types of each column and non-null value counts.
4.  **Descriptive Statistics**: `df.describe(include='all')` is used to generate descriptive statistics for all columns (both numerical and categorical), providing insights into central tendencies, dispersion, and unique value counts for categorical features.
5.  **Missing Values Check**: The number and percentage of missing values per column are calculated and displayed (`df.isnull().sum()`).
6.  **Label Distribution**: The distribution of the target variable (typically 'type' or 'label' in the ToN_IoT dataset) is examined using `value_counts()` to understand the frequency of each class (normal vs. different types of attacks). This is crucial for identifying class imbalance.

### Placeholder for Detailed Findings
Detailed findings from the EDA (e.g., specific feature distributions, correlations between features, detailed analysis of class imbalance, identification of potentially problematic or highly informative features) should be documented here after running the `01_initial_data_exploration.ipynb` notebook with the actual ToN_IoT dataset. This section will be updated once the dataset is acquired and the notebook is fully executed and analyzed.
This will include:
*   Key observations on data quality.
*   Insights into feature characteristics.
*   Confirmation of class imbalance and its extent.
*   Potential data preprocessing steps identified during EDA.
