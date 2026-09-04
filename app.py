import os
import sys

import streamlit as st
import pandas as pd
import plotly.express as px


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="EV Diagnostic Hub",
    page_icon="⚡",
    layout="wide"
)


# ============================================================
# PATH CONFIGURATION
# ============================================================

CURRENT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)


DATA_PATH = os.path.join(
    CURRENT_DIR,
    "MEGANE_E_TECH_EV60_220_driv_data.csv"
)

MANUAL_PATH = os.path.join(
    CURRENT_DIR,
    "BMWi3-owners-manual.pdf"
)

VECTOR_DB_PATH = os.path.join(
    CURRENT_DIR,
    "data",
    "vector_db"
)


# ============================================================
# FILE CHECKS
# ============================================================

if os.path.exists(MANUAL_PATH):
    print(
        f"SUCCESS: BMW manual found at {MANUAL_PATH}"
    )
else:
    print(
        f"WARNING: BMW manual not found at {MANUAL_PATH}"
    )


if os.path.exists(DATA_PATH):
    print(
        f"SUCCESS: Telemetry CSV found at {DATA_PATH}"
    )
else:
    print(
        f"WARNING: Telemetry CSV not found at {DATA_PATH}"
    )


# ============================================================
# IMPORT CUSTOM MODULES
# ============================================================

try:

    from etl import TelemetryETL
    from feature_store import FeatureStore
    from anomaly_model import AnomalyDetector

    from embedder import DocumentEmbedder
    from vector_store import VectorDatabase

    from prompt_orchestrator import PromptOrchestrator
    from llm_connector import LLMConnector

except ImportError as e:

    st.error(
        f"❌ Could not import one of the project modules:\n\n{e}"
    )

    st.info(
        f"Make sure these Python files are inside:\n\n"
        f"{CURRENT_DIR}"
    )

    st.stop()


# ============================================================
# RESOURCE INITIALIZATION
# ============================================================

@st.cache_resource
def load_rag_system():

    """
    Load the embedding model, ChromaDB and Gemini connector
    once and reuse them across Streamlit reruns.
    """

    print("Loading RAG system...")

    embedder = DocumentEmbedder()

    vector_db = VectorDatabase(
        db_path=VECTOR_DB_PATH
    )

    llm = LLMConnector(
        model_name="gemini-3.6-flash"
    )

    orchestrator = PromptOrchestrator()

    return (
        embedder,
        vector_db,
        llm,
        orchestrator
    )


try:

    (
        embedder,
        vdb,
        llm,
        orchestrator
    ) = load_rag_system()

except Exception as e:

    st.error(
        f"❌ Failed to initialize RAG system:\n\n{e}"
    )

    st.stop()


# ============================================================
# TELEMETRY PIPELINE
# ============================================================

@st.cache_data
def run_pipeline():

    """
    Runs:

    CSV
      ↓
    ETL
      ↓
    Feature Engineering
      ↓
    Anomaly Detection
    """

    # --------------------------------------------------------
    # 1. ETL
    # --------------------------------------------------------

    processed_path = os.path.join(
        CURRENT_DIR,
        "data",
        "processed",
        "clean.csv"
    )

    os.makedirs(
        os.path.dirname(processed_path),
        exist_ok=True
    )

    etl = TelemetryETL(
        DATA_PATH,
        processed_path
    )

    df_clean = etl.run()

    # --------------------------------------------------------
    # 2. FEATURE ENGINEERING
    # --------------------------------------------------------

    feature_path = os.path.join(
        CURRENT_DIR,
        "data",
        "features",
        "feat.csv"
    )

    os.makedirs(
        os.path.dirname(feature_path),
        exist_ok=True
    )

    fs = FeatureStore(
        processed_path,
        feature_path
    )

    df_features = fs.run()

    # --------------------------------------------------------
    # 3. ANOMALY DETECTION
    # --------------------------------------------------------

    model_path = os.path.join(
        CURRENT_DIR,
        "models",
        "forest.joblib"
    )

    ad = AnomalyDetector(
        feature_path,
        model_path
    )

    df_results = ad.run()

    return df_results


# ============================================================
# RUN TELEMETRY PIPELINE
# ============================================================

try:

    df_results = run_pipeline()

except Exception as e:

    st.error(
        f"❌ Telemetry pipeline failed:\n\n{e}"
    )

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚡ EV Diagnostic Hub")

mode = st.sidebar.radio(
    "Navigate to:",
    [
        "Dashboard",
        "Technical Manual RAG"
    ]
)


# ------------------------------------------------------------
# Re-run pipeline button
# ------------------------------------------------------------

if st.sidebar.button(
    "🔄 Re-run Data Pipeline"
):

    st.cache_data.clear()

    st.rerun()


# ============================================================
# DASHBOARD
# ============================================================

if mode == "Dashboard":

    st.title(
        "⚡ Vehicle Telemetry & Anomaly Detection"
    )

    st.caption(
        "Renault Megane E-Tech EV60 telemetry analysis"
    )

    # ========================================================
    # BASIC SAFETY CHECK
    # ========================================================

    if df_results.empty:

        st.warning(
            "No telemetry data is available."
        )

        st.stop()


    # ========================================================
    # METRICS
    # ========================================================

    total_data_points = len(
        df_results
    )

    total_anomalies = int(
        df_results["is_anomaly"].sum()
    )

    average_voltage = (
        df_results["Ubat_ev"].mean()
    )

    m1, m2, m3 = st.columns(3)

    m1.metric(
        "Total Data Points",
        f"{total_data_points:,}"
    )

    m2.metric(
        "Anomalies Detected",
        f"{total_anomalies:,}"
    )

    m3.metric(
        "Avg. Battery Voltage",
        f"{average_voltage:.2f} V"
    )


    # ========================================================
    # TELEMETRY GRAPH
    # ========================================================

    st.subheader(
        "📈 Live Telemetry Stream"
    )

    fig = px.line(
        df_results,
        x="time",
        y="Ubat_ev",
        title="Battery Voltage Over Time"
    )

    # --------------------------------------------------------
    # Overlay anomalies
    # --------------------------------------------------------

    anomalies = df_results[
        df_results["is_anomaly"] == True
    ]

    if not anomalies.empty:

        fig.add_scatter(
            x=anomalies["time"],
            y=anomalies["Ubat_ev"],
            mode="markers",
            marker=dict(
                color="red",
                size=8
            ),
            name="Anomaly"
        )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # ========================================================
    # DIAGNOSTIC SUMMARY
    # ========================================================

    if total_anomalies > 0:

        st.warning(
            f"⚠️ System Alert: "
            f"{total_anomalies:,} anomalies detected."
        )


        # ----------------------------------------------------
        # Anomaly data
        # ----------------------------------------------------

        anomaly_df = df_results[
            df_results["is_anomaly"] == True
        ].copy()


        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        max_current = None
        lowest_efficiency = None
        peak_time = None


        if (
            "I_bat_ev" in anomaly_df.columns
            and not anomaly_df.empty
        ):

            max_current = anomaly_df[
                "I_bat_ev"
            ].max()


        if (
            "efficiency_ratio" in anomaly_df.columns
            and not anomaly_df.empty
        ):

            lowest_efficiency = anomaly_df[
                "efficiency_ratio"
            ].min()


        if (
            "anomaly_magnitude" in anomaly_df.columns
            and not anomaly_df.empty
        ):

            peak_row = anomaly_df.loc[
                anomaly_df[
                    "anomaly_magnitude"
                ].idxmax()
            ]

            peak_time = peak_row["time"]


        # ----------------------------------------------------
        # Executive summary
        # ----------------------------------------------------

        st.subheader(
            "🔎 Diagnostic Executive Summary"
        )


        if max_current is not None:

            st.write(
                f"**Peak Current:** "
                f"{max_current:.2f} A"
            )


        if lowest_efficiency is not None:

            st.write(
                f"**Lowest Efficiency:** "
                f"{lowest_efficiency * 100:.1f}%"
            )


        if peak_time is not None:

            st.write(
                f"**Major Anomaly:** "
                f"{peak_time}s"
            )


        # ====================================================
        # TECHNICAL LOG
        # ====================================================

        with st.expander(
            "📋 View Technical Timestamp Logs"
        ):

            available_columns = [
                column
                for column in [
                    "time",
                    "Ubat_ev",
                    "I_bat_ev",
                    "efficiency_ratio"
                ]
                if column in anomaly_df.columns
            ]


            if available_columns:

                technical_logs = anomaly_df[
                    available_columns
                ].head(20).copy()


                rename_map = {
                    "time": "Time (s)",
                    "Ubat_ev": "Voltage (V)",
                    "I_bat_ev": "Current (A)",
                    "efficiency_ratio": "Efficiency"
                }


                technical_logs.rename(
                    columns=rename_map,
                    inplace=True
                )


                if "Efficiency" in technical_logs.columns:

                    technical_logs[
                        "Efficiency"
                    ] = (
                        technical_logs[
                            "Efficiency"
                        ] * 100
                    ).round(2)


                st.dataframe(
                    technical_logs,
                    use_container_width=True
                )

            else:

                st.info(
                    "No technical anomaly fields available."
                )


# ============================================================
# TECHNICAL MANUAL RAG
# ============================================================

elif mode == "Technical Manual RAG":

    st.title(
        "🤖 AI Service Assistant"
    )

    st.info(
        "Ask questions about the BMW i3 owner's manual "
        "or the Renault telemetry data."
    )


    # ========================================================
    # CHAT HISTORY
    # ========================================================

    if "messages" not in st.session_state:

        st.session_state.messages = []


    # --------------------------------------------------------
    # Display previous messages
    # --------------------------------------------------------

    for message in st.session_state.messages:

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message["content"]
            )


    # ========================================================
    # CHAT INPUT
    # ========================================================

    user_prompt = st.chat_input(
        "Ask a question..."
    )


    if user_prompt:

        # ----------------------------------------------------
        # Store user message
        # ----------------------------------------------------

        st.session_state.messages.append(
            {
                "role": "user",
                "content": user_prompt
            }
        )


        # ----------------------------------------------------
        # Display user message
        # ----------------------------------------------------

        with st.chat_message("user"):

            st.markdown(
                user_prompt
            )


        # ====================================================
        # ASSISTANT
        # ====================================================

        with st.chat_message("assistant"):

            try:

                with st.spinner(
                    "Searching the BMW manual and telemetry..."
                ):

                    # ========================================
                    # 1. EMBED USER QUESTION
                    # ========================================

                    query_vector = (
                        embedder.embed_query(
                            user_prompt
                        )
                    )


                    # ========================================
                    # 2. SEARCH CHROMADB
                    # ========================================

                    manual_context = vdb.query(
                        query_vector,
                        n_results=3
                    )


                    # ========================================
                    # 3. TELEMETRY CONTEXT
                    # ========================================

                    if (
                        df_results is not None
                        and not df_results.empty
                        and "is_anomaly" in df_results.columns
                    ):

                        telemetry_context = (
                            df_results[
                                df_results[
                                    "is_anomaly"
                                ] == True
                            ]
                            .head(20)
                        )

                    else:

                        telemetry_context = (
                            pd.DataFrame()
                        )


                    # ========================================
                    # 4. CREATE GROUNDED PROMPT
                    # ========================================

                    final_prompt = (
                        orchestrator.create_diagnostic_prompt(
                            user_prompt,
                            telemetry_context,
                            manual_context
                        )
                    )


                    # ========================================
                    # 5. CALL GEMINI
                    # ========================================

                    response = llm.get_response(
                        final_prompt
                    )


                # =================================================
                # DISPLAY RESPONSE
                # =================================================

                st.markdown(
                    response
                )


                # =================================================
                # RETRIEVED SOURCES
                # =================================================

                with st.expander(
                    "📚 Retrieved Manual Sources"
                ):

                    try:

                        documents = (
                            manual_context
                            .get(
                                "documents",
                                [[]]
                            )[0]
                        )

                        metadatas = (
                            manual_context
                            .get(
                                "metadatas",
                                [[]]
                            )[0]
                        )

                        distances = (
                            manual_context
                            .get(
                                "distances",
                                [[]]
                            )[0]
                        )


                        if documents:

                            for i, document in enumerate(
                                documents
                            ):

                                st.markdown(
                                    f"**Source {i + 1}**"
                                )


                                if i < len(metadatas):

                                    st.caption(
                                        f"Metadata: "
                                        f"{metadatas[i]}"
                                    )


                                if i < len(distances):

                                    st.caption(
                                        f"Similarity distance: "
                                        f"{distances[i]:.4f}"
                                    )


                                st.text(
                                    document[:1000]
                                )


                                if i < len(documents) - 1:

                                    st.divider()

                        else:

                            st.info(
                                "No manual sources were retrieved."
                            )

                    except Exception as source_error:

                        st.warning(
                            "Could not display retrieved "
                            f"sources: {source_error}"
                        )


            except Exception as e:

                response = (
                    "❌ I couldn't generate an answer.\n\n"
                    f"Error: `{str(e)}`"
                )

                st.error(
                    response
                )


        # ----------------------------------------------------
        # Save assistant response
        # ----------------------------------------------------

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response
            }
        )