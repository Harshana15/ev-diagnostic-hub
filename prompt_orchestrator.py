class PromptOrchestrator:

    def create_diagnostic_prompt(
        self,
        user_question,
        telemetry_context,
        manual_context
    ):
        """
        Create a concise grounded prompt.

        The LLM answers the user's actual question.
        Telemetry is included as supporting evidence,
        not as an instruction to create a diagnostic report.
        """

        # ==================================================
        # MANUAL CONTEXT
        # ==================================================

        manual_text = ""

        try:

            documents = manual_context.get(
                "documents",
                [[]]
            )[0]

            metadatas = manual_context.get(
                "metadatas",
                [[]]
            )[0]

            if documents:

                for i, document in enumerate(documents):

                    manual_text += (
                        f"\n--- BMW i3 MANUAL SOURCE {i + 1} ---\n"
                    )

                    if i < len(metadatas):
                        manual_text += (
                            f"Metadata: {metadatas[i]}\n"
                        )

                    manual_text += (
                        f"{document}\n"
                    )

            else:

                manual_text = "No manual evidence retrieved."

        except Exception:

            manual_text = "No manual evidence retrieved."

        # ==================================================
        # TELEMETRY CONTEXT
        # ==================================================

        telemetry_text = ""

        try:

            if (
                telemetry_context is not None
                and not telemetry_context.empty
            ):

                telemetry_text = telemetry_context.to_string(
                    index=False
                )

            else:

                telemetry_text = "No telemetry evidence available."

        except Exception:

            telemetry_text = "No telemetry evidence available."

        # ==================================================
        # FINAL PROMPT
        # ==================================================

        prompt = f"""
You are a concise EV technical assistant.

Answer the user's question directly.

USER QUESTION:
{user_question}

BMW i3 MANUAL EVIDENCE:
{manual_text}

RENAULT MEGANE E-TECH TELEMETRY:
{telemetry_text}

IMPORTANT:

The BMW i3 manual and Renault Megane E-Tech telemetry
are from different vehicles.

Do not assume BMW information applies to Renault.

Use only the supplied evidence.

Do not use outside knowledge to fill missing information.

Do not invent facts, specifications, page numbers,
diagnostic codes, component failures, or procedures.

If the manual does not contain enough information,
say:

"The retrieved BMW i3 manual context does not provide
enough information to answer this question."

If the telemetry does not contain enough information,
say:

"The available telemetry does not provide enough
information to answer this question."

Simple mathematical calculations using supplied telemetry
are allowed.

Clearly distinguish calculations and interpretations
from facts.

If you make an interpretation, begin it with:

"Diagnostic inference:"

Never claim that high current alone proves battery
overheating.

Never claim that efficiency losses are entirely battery
heat unless the supplied evidence explicitly establishes it.

Never invent cooling-system failures, contactor failures,
cell failures, short circuits, or other faults.

RESPONSE RULE:

Return ONLY the answer to the user's question.

Do not create a diagnostic report.

Do not create a report title.

Do not write "Senior EV Diagnostic Report".

Do not write "Telemetry Analysis" unless the user asked
about telemetry.

Do not write "Potential Causes" unless the user asks
about causes.

Do not write "Recommended Next Steps" unless the user
asks for recommendations.

Do not add "To:", "From:", "Date:", or "Subject:".

Do not repeat unrelated telemetry information.

Keep the answer concise.

If manual evidence is relevant, you may use:

### Answer

### Manual Evidence

### Evidence Limitation

Only include this section when there is a meaningful
limitation in the evidence relevant to the user's question.

Do not mention telemetry if the user's question is
about the BMW manual and telemetry is not relevant.

Do not mention the BMW manual if the user's question
is purely about telemetry.
"""

        return prompt