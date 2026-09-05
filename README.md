#  AI-Powered EV Diagnostic Hub

AI-powered EV diagnostic platform combining **telemetry anomaly detection, Retrieval-Augmented Generation (RAG), ChromaDB vector search, and Google Gemini** for evidence-grounded diagnostic analysis.

##  What It Does

    EV Telemetry
         ↓
    Data Cleaning & Feature Engineering
         ↓
    Isolation Forest
         ↓
    Anomaly Detection
         ↓
    Technical Document RAG
         ↓
    ChromaDB Vector Search
         ↓
    Google Gemini
         ↓
    AI-Assisted Diagnostic Response

##  Results

- **22,954** EV telemetry records processed
- **1,148** unusual records identified (~5%)
- **236-page** technical manual processed
- **543** document chunks created
- **384-dimensional** semantic embeddings
- Interactive **Streamlit + Plotly** dashboard

##  Key Components

###  Telemetry Anomaly Detection

Uses **Pandas, feature engineering, and Scikit-learn Isolation Forest** to identify unusual EV operating patterns.

Engineered features include:

- `efficiency_ratio`
- `delta_v`
- `delta_i`
- `v_i_ratio`
- Rolling power statistics
- `power_per_rpm`

###  Technical Document RAG

Technical documentation is processed through:

    PDF
     ↓
    Text Extraction
     ↓
    Chunking
     ↓
    Sentence Transformer Embeddings
     ↓
    ChromaDB
     ↓
    Semantic Retrieval

Embeddings are generated using **Sentence Transformers (`all-MiniLM-L6-v2`)**.

###  LLM Diagnostic Reasoning

Retrieved technical evidence and telemetry information are combined through a prompt-orchestration layer and passed to **Google Gemini**.

The system distinguishes between:

- Telemetry observations
- Calculations
- Retrieved documentation
- Diagnostic inference

This helps avoid unsupported fault conclusions.

##  Application

Built with **Streamlit + Plotly**.

The application provides:

- EV telemetry dashboard
- Anomaly visualization
- Diagnostic summaries
- Technical-document RAG interface
- AI-assisted diagnostic responses

##  Tech Stack

**Python | Pandas | NumPy | Scikit-learn | LangChain | Sentence Transformers | ChromaDB | Google Gemini | Streamlit | Plotly**

##  Project Structure

    ev-diagnostic-hub/
    ├── app.py
    ├── etl.py
    ├── feature_store.py
    ├── anomaly_model.py
    ├── pdf_chunker.py
    ├── embedder.py
    ├── vector_store.py
    ├── build_vector_db.py
    ├── prompt_orchestrator.py
    ├── llm_connector.py
    ├── event_store.py
    ├── check_models.py
    └── models/
        └── forest.joblib

##  Scope & Limitations

This is an **AI-assisted diagnostic prototype**, not a production vehicle diagnostic system.

The telemetry dataset represents a **Renault Megane E-Tech**, while the current RAG prototype uses a **BMW i3 technical manual**. The BMW documentation should therefore not be interpreted as vehicle-specific documentation for the Renault telemetry.

Detected anomalies indicate unusual patterns, not confirmed physical faults.

##  Future Improvements

- Vehicle-specific technical documentation
- Multi-document RAG
- Battery State-of-Health estimation
- Thermal anomaly detection
- Real-time telemetry / CAN-bus integration
- Supervised fault classification

##  Author

**Harshana**

GitHub: https://github.com/Harshana15

Project Repository: https://github.com/Harshana15/ev-diagnostic-hub
