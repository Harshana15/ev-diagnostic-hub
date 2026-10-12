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

## MCP Integration

The EV Diagnostic Hub exposes its diagnostic capabilities through the Model Context Protocol (MCP), allowing compatible MCP clients to interact with the telemetry analysis and technical-document retrieval pipeline.

### Available MCP Tools

- `get_vehicle_summary` — Returns a high-level overview of the diagnostic system.
- `detect_vehicle_anomalies` - Reports the total telemetry records, potential anomalies, and anomaly percentage.
- `search_technical_manual` - Searches the indexed technical manual using semantic retrieval.
- `investigate_vehicle` - Combines telemetry anomaly analysis with relevant technical-document evidence.
- `ask_diagnostic_question` - Uses Google Gemini to generate an evidence-grounded response from telemetry and retrieved manual context.

### Setup and Testing

1. Create and activate a Python virtual environment.
2. Install the dependencies listed in `requirements.txt`.
3. Configure the Google Gemini API key in the local `.env` file, following the configuration expected by `llm_connector.py`.
4. Ensure the processed telemetry results and persistent ChromaDB vector database have been generated or are available locally.
5. Start the MCP Inspector with:

   ```powershell
   mcp dev mcp_server.py
   ```

6. Connect to the server and test the available tools in the Inspector.

Do not commit API keys, `.env` files, or generated data and model artifacts unless the project explicitly requires them.

### MCP Diagnostic Limitations

The current telemetry dataset represents a Renault Megane E-Tech, while the indexed technical manual is for a BMW i3. Retrieved BMW documentation is not vehicle-specific guidance for the Renault dataset.

Anomaly detection identifies unusual statistical patterns; it does not independently establish a physical fault. Likewise, high current or calculated power loss alone does not confirm battery overheating without supporting temperature evidence.

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
