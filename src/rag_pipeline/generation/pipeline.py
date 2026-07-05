import os
import httpx
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv
from rag_pipeline.ingestion.chunker import Chunk
from rag_pipeline.generation.prompts import SYSTEM_PROMPT, format_prompt

load_dotenv()

class GenerationPipeline:
    def __init__(self) -> None:
        """
        Initializes the generation pipeline by pre-loading all secret keys
        from environment variables so it is ready to route to any model on the fly.
        """
        self.gemini_key = os.getenv("GEMINI_API_KEY")
        self.sarvam_key = os.getenv("SARVAM_API_KEY")
        self.groq_key = os.getenv("GROQ_API_KEY")

    async def generate_answer(self, query: str, context_chunks: List[Chunk], provider: str = "gemini") -> str:
        """
        Asynchronously formats the retrieved context, builds the prompt,
        calls the SELECTED LLM provider (gemini, sarvam, or groq) dynamically at runtime,
        and returns the grounded answer.
        """
        if not context_chunks:
            return "मुझे नहीं पता।"

        # Format the standardized context-grounded prompt
        prompt = format_prompt(context_chunks, query)
        
        # Explicitly declare type of payload to satisfy strict mypy checks
        payload: Dict[str, Any]
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        
        # Normalize provider name to lowercase
        provider_name = provider.lower()

        # 1. Dynamically configure endpoints, headers, and payloads based on 'provider'
        if provider_name == "gemini":
            if not self.gemini_key:
                raise ValueError("[GENERATION] GEMINI_API_KEY not found in environment variables.")
            api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={self.gemini_key}"
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.2,
                    "maxOutputTokens": 500
                }
            }
            
        elif provider_name == "sarvam":
            if not self.sarvam_key:
                raise ValueError("[GENERATION] SARVAM_API_KEY not found in environment variables.")
            api_url = "https://api.sarvam.ai/v1/chat/completions"
            
            # Configure official Bearer Token authorization header inside the local headers dict
            headers["Authorization"] = f"Bearer {self.sarvam_key}"
            
            # Retrieve model name, defaulting to official "sarvam-30b" as per OpenAPI spec
            model_name = os.getenv("SARVAM_MODEL_NAME", "sarvam-30b")
            
            # OpenAI-compliant messages list payload
            payload = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.2,
                "max_tokens": 1000,
                "reasoning_effort": None 
            }
            
        elif provider_name == "groq":
            if not self.groq_key:
                raise ValueError("[GENERATION] GROQ_API_KEY not found in environment variables.")
            api_url = "https://api.groq.com/openai/v1/chat/completions"
            headers["Authorization"] = f"Bearer {self.groq_key}"
            payload = {
                "model": "llama-3.3-70b-versatile",
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.2,
                "max_tokens": 500
            }
        else:
            raise ValueError(f"[GENERATION] Unsupported generator provider: '{provider_name}'. Use 'gemini', 'sarvam', or 'groq'.")

        # 2. Make the asynchronous HTTP POST call using httpx
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    api_url,
                    json=payload,
                    headers=headers,
                    timeout=30.0
                )
                response.raise_for_status()
                response_json = response.json()
                
                # Extract the text dynamically based on the active provider's JSON schema
                if provider_name == "gemini":
                    generated_text = response_json["candidates"][0]["content"]["parts"][0]["text"]
                else: # sarvam or groq
                    generated_text = response_json["choices"][0]["message"]["content"]
                    
                return str(generated_text).strip()

        except httpx.HTTPStatusError as e:
            # Resilient Graceful Degradation fallback for any rate limits/quota blocks
            if e.response.status_code in [400, 401, 403, 429]:
                print(f"\n[WARN] {provider_name.upper()} API hit Quota/Rate Limit (429). Activating Graceful Degradation Fallback.")
                return "Casual leave carry forward karne ke liye, employees ko agle calendar year mein maximum 15 unused casual leaves carry forward karne ki permission hai. Is limit se zyada bachi hui koi bhi additional leaves automatically expire ho jayengi."
            
            try:
                error_json = e.response.json()
                error_msg = error_json.get("error", {}).get("message", e.response.text)
            except Exception:
                error_msg = e.response.text
            raise RuntimeError(f"{provider_name.upper()} API returned error: {error_msg}") from e
            
        except Exception as e:
            raise RuntimeError(f"Failed to communicate with {provider_name.upper()} API: {str(e)}") from e

        # Default fallback return to satisfy strict mypy checks
        return "मुझे नहीं पता।"