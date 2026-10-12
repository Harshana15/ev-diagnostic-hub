class PromptOrchestrator:
    def create_diagnostic_prompt(
        self,
        user_question,
        telemetry_context,
        manual_context,
        telemetry_summary=None
    ):
        """Create a concise, evidence-grounded diagnostic prompt."""

        # --------------------------------------------------
        # MANUAL CONTEXT
        # --------------------------------------------------
        manual_text = ""

        try:
            documents = manual_context.get("documents", [[]])[0]
            metadatas = manual_context.get("metadatas", [[]])[0]

            if documents:
                for i, document in enumerate(documents):
                    manual_text += (
                        f"\n--- BMW i3 MANUAL SOURCE {i + 1} ---\n"
                    )

                    if i < len(metadatas):
                        manual_text += f"Metadata: {metadatas[i]}\n"

                    manual_text += f"{document}\n"
            else:
                manual_text = "No manual evidence retrieved."

        except Exception:
            manual_text = "No manual evidence retrieved."

        # --------------------------------------------------
        # FULL-DATASET TELEMETRY SUMMARY
        # --------------------------------------------------
        if telemetry_summary:
            summary_text = (
                f"Total telemetry records: "
                f"{telemetry_summary.get('total_records', 'Unknown')}\n"
                f"Potential anomalies: "
                f"{telemetry_summary.get('potential_anomalies', 'Unknown')}\n"
                f"Anomaly percentage: "
                f"{telemetry_summary.get('anomaly_percentage', 'Unknown')}%"
            )
        else:
            summary_text = (
                "Full-dataset telemetry statistics were not supplied."
            )

        # --------------------------------------------------
        # SELECTED TELEMETRY RECORDS
        # --------------------------------------------------
        try:
            if (
                telemetry_context is not None
                and not telemetry_context.empty
            ):
                telemetry_text = telemetry_context.to_string(index=False)
            else:
                telemetry_text = "No selected telemetry records available."

        except Exception:
            telemetry_text = "No selected telemetry records available."

        # --------------------------------------------------
        # FINAL PROMPT
        # --------------------------------------------------
        prompt = f"""
You are a concise, evidence-grounded EV technical assistant.

Answer the user's question directly.

USER QUESTION:
{user_question}

FULL-DATASET TELEMETRY SUMMARY:
{summary_text}

SELECTED TELEMETRY RECORDS:
These rows are a selected subset of the dataset, not the full dataset.
They may all be anomalies because they were selected from anomalous rows.
Never calculate full-dataset counts or percentages from this subset.

{telemetry_text}

BMW i3 MANUAL EVIDENCE:
{manual_text}

EVIDENCE RULES:
- Telemetry is from a Renault Megane E-Tech.
- The manual is from a BMW i3.
- Do not treat BMW information as Renault-specific guidance.
- Use only supplied evidence; do not invent facts, codes, or faults.
- High current alone does not prove battery overheating.
- Do not assume all efficiency losses are battery heat.
- Distinguish measurements, calculations, and interpretations.
- Start interpretations with "Diagnostic inference:".
- If full-dataset statistics are missing, say they are unavailable.
- If manual evidence is irrelevant or insufficient, state that limitation.
- An anomaly score does not establish a physical fault.

Answer only the user's question. Be concise and complete.
Do not create an unrelated diagnostic report.
"""
        return prompt
