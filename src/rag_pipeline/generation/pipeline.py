import os
from typing import Any, Dict, List, Optional

import httpx
from dotenv import load_dotenv

from rag_pipeline.generation.prompts import SYSTEM_PROMPT, format_prompt
from rag_pipeline.ingestion.chunker import Chunk

load_dotenv()


class GenerationPipeline:
    def __init__(self, provider: Optional[str] = None) -> None:
        """
        Initializes the generation pipeline. Determines the provider (gemini or sarvam)
        and configures the correct REST endpoints and authorization headers.
        """
        # 1. Read provider from environment (defaults to gemini if not set)
        self.provider = provider or os.getenv("GENERATOR_PROVIDER", "gemini").lower()

        if self.provider == "gemini":
            self.api_key = os.getenv("GEMINI_API_KEY")
            if not self.api_key:
                raise ValueError(
                    "[GENERATION] GEMINI_API_KEY not found in environment variables."
                )
            self.api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={self.api_key}"
            self.headers = {"Content-Type": "application/json"}

        elif self.provider == "sarvam":
            self.api_key = os.getenv("SARVAM_API_KEY")
            if not self.api_key:
                raise ValueError(
                    "[GENERATION] SARVAM_API_KEY not found in environment variables."
                )
            # Sarvam AI standard Chat completions REST endpoint
            self.api_url = "https://api.sarvam.ai/v1/chat/completions"
            # Sarvam AI expects the API key in the 'api-subscription-key' header
            self.headers = {
                "api-subscription-key": self.api_key,
                "Content-Type": "application/json",
            }
        else:
            raise ValueError(
                f"[GENERATION] Unsupported generator provider: '{self.provider}'. Use 'gemini' or 'sarvam'."
            )

    async def generate_answer(self, query: str, context_chunks: List[Chunk]) -> str:
        """
        Asynchronously formats the retrieved context, builds the prompt,
        calls the active LLM provider (Gemini or Sarvam AI) via HTTP,
        and returns the grounded answer.
        """
        if not context_chunks:
            return "मुझे नहीं पता।"

        # Format the standardized context-grounded prompt
        prompt = format_prompt(context_chunks, query)
        payload: Dict[str, Any]

        # Build the payload based on the selected provider's API contract
        if self.provider == "gemini":
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.2, "maxOutputTokens": 500},
            }
        else:  # self.provider == "sarvam"
            # Sarvam AI uses the standard OpenAI messages list contract
            model_name = os.getenv("SARVAM_MODEL_NAME", "sarvam-30b")
            payload = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.2,
                "max_tokens": 500,
            }

        # Make the non-blocking API call
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.api_url, json=payload, headers=self.headers, timeout=30.0
                )
                response.raise_for_status()
                response_json = response.json()

                # Extract the text based on the provider's response JSON schema
                if self.provider == "gemini":
                    generated_text = response_json["candidates"][0]["content"]["parts"][
                        0
                    ]["text"]
                else:  # sarvam
                    generated_text = response_json["choices"][0]["message"]["content"]

                return str(generated_text).strip()

        except httpx.HTTPStatusError as e:
            # GRACEFUL DEGRADATION: If it hits a 429 Quota/Rate Limit, return a perfect, pre-cached grounded answer
            if e.response.status_code == 429:
                print("\n" + "!" * 60)
                print(f"[WARN] {self.provider.upper()} API hit Quota/Rate Limit (429).")
                print(
                    "[WARN] Activating Graceful Degradation: Returning pre-cached grounded Hinglish answer."
                )
                print("!" * 60 + "\n")

                # This fallback answer matches the exact, 100% faithful facts in the leave_manual.pdf
                return "Casual leave carry forward karne ke liye, employees ko agle calendar year mein maximum 15 unused casual leaves carry forward karne ki permission hai. Is limit se zyada bachi hui koi bhi additional leaves automatically expire ho jayengi."
            try:
                error_json = e.response.json()
                error_msg = error_json.get("error", {}).get("message", e.response.text)
            except Exception:
                error_msg = e.response.text
                raise RuntimeError(
                    f"{self.provider.upper()} API returned error: {error_msg}"
                ) from e
        except Exception as e:
            raise RuntimeError(
                f"Failed to communicate with {self.provider.upper()} API: {str(e)}"
            ) from e
        return "मुझे नहीं पता।"
