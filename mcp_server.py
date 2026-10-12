import sys
from pathlib import Path

# ============================================================
# WINDOWS UTF-8 FIX
# ============================================================

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


# ============================================================
# MCP
# ============================================================

from mcp.server import MCPServer

mcp = MCPServer("EV Diagnostic Hub")


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

ANOMALY_RESULTS_PATH = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "telemetry_results.csv"
)

VECTOR_DB_PATH = (
    PROJECT_ROOT
    / "data"
    / "vector_db"
)


# ============================================================
# TOOL 1: VEHICLE SUMMARY
# ============================================================

@mcp.tool()
def get_vehicle_summary() -> str:
    """
    Return a basic summary of the EV Diagnostic Hub system.
    """

    return (
        "EV Diagnostic Hub is an AI-assisted EV diagnostic system "
        "that combines telemetry processing, anomaly detection, "
        "technical-document retrieval, and LLM reasoning."
    )


# ============================================================
# TOOL 2: ANOMALY DETECTION
# ============================================================

@mcp.tool()
def detect_vehicle_anomalies() -> str:
    """
    Return a summary of anomalies detected in EV telemetry.
    """

    try:
        import pandas as pd

        if not ANOMALY_RESULTS_PATH.exists():
            return (
                "Anomaly results file not found at:\n"
                f"{ANOMALY_RESULTS_PATH}"
            )

        df = pd.read_csv(ANOMALY_RESULTS_PATH)

        if df.empty:
            return "The anomaly results file contains no telemetry records."

        anomaly_count = int(df["is_anomaly"].sum())
        total_rows = len(df)

        anomaly_percentage = (
            anomaly_count / total_rows
        ) * 100

        return (
            "EV anomaly detection results:\n"
            f"Total telemetry records: {total_rows}\n"
            f"Potential anomalies: {anomaly_count}\n"
            f"Anomaly percentage: {anomaly_percentage:.2f}%"
        )

    except Exception as e:

        return (
            "Anomaly detection failed.\n"
            f"Error: {str(e)}"
        )


# ============================================================
# RAG COMPONENTS
# ============================================================

_vector_db = None
_embedder = None


def get_rag_components():
    """
    Lazily initialize ChromaDB and the embedding model.

    These imports are intentionally inside the function so that
    MCP Inspector can connect quickly without waiting for
    PyTorch/SentenceTransformers during server startup.
    """

    global _vector_db
    global _embedder

    if _vector_db is None:

        from vector_store import VectorDatabase

        _vector_db = VectorDatabase(
            db_path=str(VECTOR_DB_PATH)
        )

    if _embedder is None:

        from embedder import DocumentEmbedder

        _embedder = DocumentEmbedder(
            model_name="all-MiniLM-L6-v2"
        )

    return _vector_db, _embedder


# ============================================================
# TOOL 3: TECHNICAL MANUAL SEARCH
# ============================================================

@mcp.tool()
def search_technical_manual(
    query: str,
    n_results: int = 5
) -> str:
    """
    Search the indexed technical manual using semantic retrieval.

    The currently indexed documentation is the BMW i3 owner's
    manual. It is not vehicle-specific to the Renault Megane E-Tech
    telemetry dataset.
    """

    if not query or not query.strip():

        return (
            "Please provide a technical question or search query."
        )

    try:

        n_results = max(
            1,
            min(int(n_results), 5)
        )

        vector_db, embedder = get_rag_components()

        query_embedding = embedder.embed_query(
            query.strip()
        )

        search_results = vector_db.query(
            query_embedding,
            n_results=n_results
        )

        documents = search_results.get(
            "documents",
            [[]]
        )[0]

        metadatas = search_results.get(
            "metadatas",
            [[]]
        )[0]

        distances = search_results.get(
            "distances",
            [[]]
        )[0]

        if not documents:

            return (
                "No relevant technical documentation was found."
            )

        response = []

        response.append(
            "Technical manual search results:"
        )

        response.append("")

        response.append(
            "Source: BMW i3 owner's manual"
        )

        response.append(
            "Important: This documentation is not "
            "vehicle-specific to the Renault Megane E-Tech "
            "telemetry dataset."
        )

        response.append("")

        for i, document in enumerate(documents):

            response.append(
                f"--- Result {i + 1} ---"
            )

            if i < len(distances):

                response.append(
                    f"Distance: {distances[i]:.4f}"
                )

            if i < len(metadatas):

                response.append(
                    f"Metadata: {metadatas[i]}"
                )

            response.append(
                f"Document:\n{document}"
            )

            response.append("")

        return "\n".join(response)

    except Exception as e:

        return (
            "Technical manual search failed.\n"
            f"Error: {str(e)}"
        )


# ============================================================
# TOOL 4: COMBINED VEHICLE INVESTIGATION
# ============================================================

@mcp.tool()
def investigate_vehicle(
    query: str
) -> str:
    """
    Perform a combined EV diagnostic investigation.

    Combines:
    1. Telemetry anomaly detection
    2. Identification of highly anomalous records
    3. Semantic search of the technical manual

    The indexed technical documentation is the BMW i3 owner's
    manual and is not vehicle-specific to the Renault Megane E-Tech.
    """

    if not query or not query.strip():

        return "Please provide a diagnostic question."

    try:

        import pandas as pd

        # ----------------------------------------------------
        # LOAD TELEMETRY RESULTS
        # ----------------------------------------------------

        if not ANOMALY_RESULTS_PATH.exists():

            return (
                "Diagnostic investigation failed.\n"
                "Telemetry anomaly results file was not found."
            )

        df = pd.read_csv(
            ANOMALY_RESULTS_PATH
        )

        if df.empty:

            return (
                "Diagnostic investigation failed.\n"
                "Telemetry results contain no records."
            )

        anomaly_df = df[
            df["is_anomaly"] == True
        ].copy()

        total_records = len(df)
        anomaly_count = len(anomaly_df)

        anomaly_percentage = (
            anomaly_count / total_records
        ) * 100

        # ----------------------------------------------------
        # MOST ANOMALOUS RECORDS
        # ----------------------------------------------------

        top_anomalies = anomaly_df.sort_values(
            by="anomaly_magnitude",
            ascending=True
        ).head(5)

        # ----------------------------------------------------
        # MANUAL RETRIEVAL
        # ----------------------------------------------------

        vector_db, embedder = get_rag_components()

        query_embedding = embedder.embed_query(
            query.strip()
        )

        search_results = vector_db.query(
            query_embedding,
            n_results=3
        )

        documents = search_results.get(
            "documents",
            [[]]
        )[0]

        metadatas = search_results.get(
            "metadatas",
            [[]]
        )[0]

        distances = search_results.get(
            "distances",
            [[]]
        )[0]

        # ----------------------------------------------------
        # BUILD RESULT
        # ----------------------------------------------------

        response = []

        response.append(
            "EV Diagnostic Investigation"
        )

        response.append(
            "============================"
        )

        response.append("")

        response.append(
            "Diagnostic question:"
        )

        response.append(
            query.strip()
        )

        response.append("")

        response.append(
            "TELEMETRY ANOMALY SUMMARY"
        )

        response.append(
            "--------------------------"
        )

        response.append(
            f"Total telemetry records: {total_records}"
        )

        response.append(
            f"Potential anomalies: {anomaly_count}"
        )

        response.append(
            f"Anomaly percentage: {anomaly_percentage:.2f}%"
        )

        response.append("")

        response.append(
            "MOST ANOMALOUS TELEMETRY RECORDS"
        )

        response.append(
            "--------------------------------"
        )

        for index, row in top_anomalies.iterrows():

            response.append(
                f"Record index: {index}"
            )

            for column in [
                "time",
                "I_bat_ev",
                "Ubat_ev",
                "Power_DC_ev",
                "efficiency_ratio",
                "v_i_ratio",
                "rolling_power_std",
                "power_per_rpm",
                "anomaly_magnitude"
            ]:

                if column in row:

                    response.append(
                        f"  {column}: {row[column]}"
                    )

            response.append("")

        # ----------------------------------------------------
        # TECHNICAL DOCUMENTATION
        # ----------------------------------------------------

        response.append(
            "TECHNICAL DOCUMENTATION EVIDENCE"
        )

        response.append(
            "--------------------------------"
        )

        response.append(
            "Source: BMW i3 owner's manual"
        )

        response.append(
            "Important: The technical documentation is "
            "not vehicle-specific to the Renault Megane "
            "E-Tech telemetry dataset."
        )

        response.append("")

        if documents:

            for i, document in enumerate(documents):

                response.append(
                    f"--- Evidence {i + 1} ---"
                )

                if i < len(distances):

                    response.append(
                        f"Distance: {distances[i]:.4f}"
                    )

                if i < len(metadatas):

                    metadata = metadatas[i]

                    response.append(
                        f"Source: {metadata.get('source', 'Unknown')}"
                    )

                    response.append(
                        f"Page: {metadata.get('page_label', 'Unknown')}"
                    )

                response.append(
                    f"Document:\n{document}"
                )

                response.append("")

        else:

            response.append(
                "No technical documentation evidence was found."
            )

            response.append("")

        # ----------------------------------------------------
        # LIMITATION
        # ----------------------------------------------------

        response.append(
            "INVESTIGATION NOTE"
        )

        response.append(
            "------------------"
        )

        response.append(
            "The anomaly detector identifies records that "
            "differ from the learned telemetry distribution."
        )

        response.append(
            "It does not by itself identify the physical "
            "cause of an anomaly."
        )

        response.append(
            "The retrieved documentation is from the BMW i3 "
            "owner's manual and must not be treated as "
            "vehicle-specific guidance for the Renault "
            "Megane E-Tech."
        )

        return "\n".join(response)

    except Exception as e:

        return (
            "Diagnostic investigation failed.\n"
            f"Error: {str(e)}"
        )


# ============================================================
# TOOL 5: ASK DIAGNOSTIC QUESTION
# ============================================================

@mcp.tool()
def ask_diagnostic_question(question: str) -> str:
    """Ask Gemini a diagnostic question using telemetry and manual context."""

    if not question or not question.strip():
        return "Please provide a diagnostic question."

    try:
        import pandas as pd
        from llm_connector import LLMConnector
        from prompt_orchestrator import PromptOrchestrator

        # 1. Load telemetry results
        if not ANOMALY_RESULTS_PATH.exists():
            return "The telemetry results file was not found."

        df = pd.read_csv(ANOMALY_RESULTS_PATH)

        # 2. Select anomalous records
        if "is_anomaly" in df.columns:
            anomaly_df = df[
                df["is_anomaly"].fillna(False).astype(bool)
            ].copy()
        else:
            anomaly_df = df.copy()

        # 3. Select up to 10 most anomalous records
        if (
            "anomaly_magnitude" in anomaly_df.columns
            and not anomaly_df.empty
        ):
            telemetry_context = anomaly_df.sort_values(
                by="anomaly_magnitude",
                ascending=True
            ).head(10)
        else:
            telemetry_context = anomaly_df.head(10)

        # 4. Calculate full-dataset statistics
        total_records = len(df)

        if "is_anomaly" in df.columns:
            anomaly_count = int(
                df["is_anomaly"].fillna(False).astype(bool).sum()
            )
        else:
            anomaly_count = 0

        telemetry_summary = {
            "total_records": total_records,
            "potential_anomalies": anomaly_count,
            "anomaly_percentage": (
                round(anomaly_count / total_records * 100, 2)
                if total_records > 0 else 0
            ),
        }

        # 5. Retrieve manual context using cached components
        vector_db, embedder = get_rag_components()
        query_embedding = embedder.embed_query(question.strip())

        manual_context = vector_db.query(
            query_embedding,
            n_results=3
        )

        # 6. Build the diagnostic prompt
        orchestrator = PromptOrchestrator()

        final_prompt = orchestrator.create_diagnostic_prompt(
            user_question=question.strip(),
            telemetry_context=telemetry_context,
            manual_context=manual_context,
            telemetry_summary=telemetry_summary
        )

        # 7. Ask Gemini
        connector = LLMConnector()
        answer = connector.get_response(final_prompt)

        return str(answer)

    except Exception as e:
        return (
            "Diagnostic question failed.\n"
            f"Error: {e}"
        )
if __name__ == "__main__":
    mcp.run()