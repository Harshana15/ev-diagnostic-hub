# AI-Powered EV Diagnostic Hub

An AI-powered Electric Vehicle diagnostic system combining **machine-learning anomaly detection**, **Retrieval-Augmented Generation (RAG)**, **vector search**, and **LLM-based reasoning** to support evidence-grounded EV diagnostic analysis.

The system processes EV telemetry data to identify anomalous operating conditions and combines relevant technical-document information with telemetry evidence to generate concise diagnostic assistance through a Streamlit interface.

---

## 🚗 Project Overview

Modern EVs generate large volumes of telemetry data, making it difficult to manually identify abnormal operating conditions and connect them with relevant technical documentation.

This project addresses that problem by combining two complementary AI pipelines:

### 1. EV Telemetry Analytics

Raw EV telemetry is processed through:

```text
Raw Telemetry
     ↓
ETL / Data Cleaning
     ↓
Feature Engineering
     ↓
Isolation Forest
     ↓
Anomaly Detection
     ↓
Streamlit Dashboard

## 2. Technical Document RAG

The Technical Document RAG pipeline enables the system to retrieve relevant information from EV technical documentation and provide evidence-grounded responses through an LLM.

### RAG Pipeline

```text
Technical Manual PDF
        ↓
PDF Text Extraction
        ↓
Recursive Character Chunking
        ↓
Sentence Transformer Embeddings
        ↓
ChromaDB Vector Store
        ↓
Semantic Retrieval
        ↓
Relevant Manual Evidence
        ↓
Google Gemini
        ↓
Evidence-Grounded Diagnostic Response
