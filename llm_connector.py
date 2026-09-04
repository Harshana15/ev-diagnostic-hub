import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI


# Load environment variables
load_dotenv(override=True)


class LLMConnector:

    def __init__(self, model_name="gemini-3.6-flash"):

        api_key = os.getenv("GOOGLE_API_KEY")

        if not api_key:
            raise ValueError(
                "❌ GOOGLE_API_KEY not found in .env. Check your file name!"
            )

        print(
            f"📡 System: Connecting to Stable Pipeline ({model_name})..."
        )

        self.model = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0.1,
            api_version="v1",
            max_output_tokens=2048
        )

    def get_response(self, final_prompt: str):

        try:

            # Send prompt to Gemini
            response = self.model.invoke(final_prompt)

            content = response.content

            # Gemini/LangChain can return structured content
            if isinstance(content, list):

                text_parts = []

                for block in content:

                    if isinstance(block, dict):

                        if block.get("type") == "text":
                            text_parts.append(
                                block.get("text", "")
                            )

                    elif isinstance(block, str):

                        text_parts.append(block)

                return "\n".join(text_parts).strip()

            # Normal string response
            return str(content)

        except Exception as e:

            return f"❌ LLM Error: {str(e)}"


# --------------------------------------------------
# CONNECTION TEST
# --------------------------------------------------

if __name__ == "__main__":

    print("🧪 Running Connection Test...")

    connector = LLMConnector()

    result = connector.get_response(
        "Say 'Connection Successful' if you can read this."
    )

    print(result)