# Complete Pipeline  

This project covers the **end-to-end understanding of a Machine Learning pipeline** and working with **DVC (Data Version Control)** for automation, reproducibility, and experiment tracking.  

---

## Project Overview  

The pipeline includes:  
1. Data Ingestion – loading raw data  
2. Data Preprocessing – cleaning, scaling, encoding  
3. Feature Engineering – transforming and selecting features  
4. Model Building – training machine learning models  
5. Model Evaluation – validating results and logging metrics  
6. Version Control with DVC – tracking datasets, experiments, and models  

---

## Key Features  

- Modular pipeline with Python scripts under `src/`  
- Parameters managed in `params.yaml` for flexibility  
- Logs stored in the `logs/` directory for debugging each stage  
- DVC integration for pipeline automation and reproducibility  
- Metrics and reports generated after evaluation  

---

## Directory Structure  

# Complete Pipeline  

This project covers the **end-to-end understanding of a Machine Learning pipeline** and working with **DVC (Data Version Control)** for automation, reproducibility, and experiment tracking.  

---

## Project Overview  

The pipeline includes:  
1. Data Ingestion – loading raw data  
2. Data Preprocessing – cleaning, scaling, encoding  
3. Feature Engineering – transforming and selecting features  
4. Model Building – training machine learning models  
5. Model Evaluation – validating results and logging metrics  
6. Version Control with DVC – tracking datasets, experiments, and models  

---

## Key Features  

- Modular pipeline with Python scripts under `src/`  
- Parameters managed in `params.yaml` for flexibility  
- Logs stored in the `logs/` directory for debugging each stage  
- DVC integration for pipeline automation and reproducibility  
- Metrics and reports generated after evaluation  

---

---

## Pipeline Flow (DVC DAG)  

```mermaid
flowchart TD
    A[Data Ingestion] --> B[Data Preprocessing]
    B --> C[Feature Engineering]
    C --> D[Model Building]
    D --> E[Model Evaluation]
    E --> F[Reports & Metrics]
