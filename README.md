# GCP FinOps Anomaly Detection Dashboard

This project is an intelligent cloud FinOps system designed to analyze GCP billing data and detect cost anomalies using a combination of machine learning and rule-based techniques. It provides an interactive dashboard for monitoring, analysis, and optimization of cloud expenses.



## Overview

The application processes cloud billing data and identifies unusual patterns such as cost spikes, abnormal usage behavior, and inefficient resource utilization. It integrates statistical analysis with machine learning to improve detection accuracy and provide actionable insights.



## Features

* Hybrid anomaly detection using Isolation Forest and rule-based logic
* Interactive dashboard for cost analysis and visualization
* Severity classification of anomalies
* Root cause identification
* Optimization recommendations with estimated savings
* ChatOps interface for querying data using natural language



## Tech Stack

* Streamlit
* Pandas
* NumPy
* Scikit-learn
* Plotly



## Project Structure

```
cloud/
│── anomaly.py
│── requirements.txt
│── README.md
```



## Input Requirements

The application expects a CSV file containing GCP billing data. The following columns are required or recommended:

* Usage Start Date
* Usage End Date (optional)
* Unrounded Cost ($)
* Usage Quantity
* Service Name
* Region/Zone
* CPU Utilization (%) (optional)
* Memory Utilization (%) (optional)



## How to Run

1. Clone the repository

```
git clone https://github.com/your-username/cloud-finops-anomaly-detection.git
cd cloud-finops-anomaly-detection
```

2. Install dependencies

```
pip install -r requirements.txt
```

3. Run the application

```
streamlit run anomaly.py
```



## Methodology

### Data Preprocessing

* Converts timestamps and extracts time-based features
* Computes cost, usage, and efficiency metrics
* Applies rolling averages for trend detection

### Machine Learning Detection

* Scales features using StandardScaler
* Uses Isolation Forest to identify anomalous patterns

### Rule-Based Detection

* Detects daily cost spikes
* Identifies service and region-level deviations
* Flags underutilized resources
* Detects abnormal usage surges

### Output Generation

* Combines ML and rule-based results
* Assigns severity levels
* Generates root cause explanations
* Provides optimization recommendations
* Estimates potential cost savings



## Output

The dashboard provides:

* Cost summary metrics
* Daily cost trends
* Service-wise cost distribution
* Detected anomalies with explanations
* Optimization suggestions and savings estimates



## Future Improvements

* Integration with cloud APIs for real-time monitoring
* Automated alerting system
* Multi-cloud support
* Enhanced conversational interface



## Author

Aswani Sachin



## License

MIT License
