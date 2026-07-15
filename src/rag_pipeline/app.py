import os
import sys
import types

# 1. Force single-threaded execution for PyTorch to prevent CPU deadlocks
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import streamlit as st

# 2. Sync Streamlit secrets into os.environ so os.getenv() calls work throughout the codebase
for key in [
    "GENERATOR_PROVIDER", "GROQ_API_KEY", "GEMINI_API_KEY", "SARVAM_API_KEY",
    "QDRANT_HOST", "QDRANT_PORT", "QDRANT_API_KEY", "HF_TOKEN"
]:
    if key in st.secrets and key not in os.environ:
        os.environ[key] = str(st.secrets[key])

# 3. Handle Ragas VertexAI import mock
if "langchain_community.chat_models.vertexai" not in sys.modules:
    mock_vertex_module = types.ModuleType("langchain_community.chat_models.vertexai")
    class DummyChatVertexAI:
        pass
    mock_vertex_module.ChatVertexAI = DummyChatVertexAI # type: ignore
    sys.modules["langchain_community.chat_models.vertexai"] = mock_vertex_module

import asyncio
import tempfile
from pathlib import Path
from dotenv import load_dotenv

# Import our modular pipeline classes
from rag_pipeline.ingestion.pipeline import IngestionPipeline
from rag_pipeline.embeddings.pipeline import EmbeddingPipeline
from rag_pipeline.retrieval.pipeline import RetrievalPipeline
from rag_pipeline.generation.pipeline import GenerationPipeline
from rag_pipeline.embeddings.model import EmbeddingModel

load_dotenv()

# Declare Global Constants
COLLECTION_NAME = "streamlit_user_manuals"

st.set_page_config(
    page_title="Indic Multilingual RAG",
    page_icon="🤖",
    layout="wide"
)

# THE SIMPLE, NON-THREADING ASYNC RUNNER
def run_async(coro):
    """
    Runs a coroutine in an isolated event loop per call.
    Perfectly safe as long as we make exactly one call per user action.
    """
    return asyncio.run(coro)

# Cache ONLY the heavy 400MB embedding model once
@st.cache_resource
def get_shared_model() -> EmbeddingModel:
    return EmbeddingModel()

shared_model = get_shared_model()

# Instantiate all lightweight pipelines fresh on every execution thread
# This ensures their Qdrant clients start fresh and bind safely to the active loop
ingest_pipeline = IngestionPipeline(chunk_size=800, chunk_overlap=150)
embed_pipeline = EmbeddingPipeline(collection_name=COLLECTION_NAME, model=shared_model)
retrieval_pipeline = RetrievalPipeline(collection_name=COLLECTION_NAME, model=shared_model)
generation_pipeline = GenerationPipeline()

# OPTION B: UNIFIED SINGLE-LOOP COROUTINES 
async def clear_and_index_file(temp_file_path: Path) -> int:
    """
    Clears the collection and indexes the new PDF in a single, isolated loop lifecycle.
    """
    # 1. Clear old collection
    await embed_pipeline.store.clear_collection(COLLECTION_NAME)
    
    # 2. Ingest and chunk new file
    chunks = await ingest_pipeline.ingest_file(temp_file_path)
    if chunks:
        # 3. Embed and store
        await embed_pipeline.embed_and_store(chunks)
        return len(chunks)
    return 0

async def run_query_pipeline(prompt_text: str, provider_key: str) -> str:
    """
    Retrieves context and generates the grounded answer in a single, isolated loop lifecycle.
    """
    chunks = await retrieval_pipeline.retrieve(query=prompt_text, top_k=2)
    if chunks:
        answer = await generation_pipeline.generate_answer(
            query=prompt_text, 
            context_chunks=chunks,
            provider=provider_key
        )
        sources_text = "\n\n**Sources:**\n" + "\n".join([f"- Page {c.metadata.page_number} ({c.metadata.source_file})" for c in chunks])
        return f"{answer}{sources_text}"
    return "मुझे नहीं पता। (No matching documents found in database)."

# Streamlit Page Layout
st.title("🤖 Multilingual Indic RAG Pipeline")
st.write("Upload an English PDF document, and ask questions about it in **Hindi, Hinglish, or English**!")

# Sidebar Configuration
with st.sidebar:
    st.header("📁 Document Ingestion")
    uploaded_file = st.file_uploader("Upload your PDF manual:", type=["pdf"])
    
    if uploaded_file is not None:
        st.info(f"File uploaded: {uploaded_file.name}")
        
        if "ingested_file" not in st.session_state or st.session_state["ingested_file"] != uploaded_file.name:
            with st.spinner("Parsing, embedding, and indexing PDF into Qdrant..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    temp_path = Path(tmp_file.name)
                
                try:
                    # Run the entire ingestion sequence inside a SINGLE loop call
                    total_chunks = run_async(clear_and_index_file(temp_path))
                    st.success(f"Success! Generated and indexed {total_chunks} chunks.")
                    st.session_state["ingested_file"] = uploaded_file.name
                except Exception as e:
                    st.error(f"Ingestion failed: {str(e)}")
                finally:
                    # Secure Cleanup: Ensure the temp file is deleted even if the code crashes
                    if temp_path.exists():
                        os.unlink(temp_path)

    # Dynamically mapped model configuration
    provider_to_label = {
        "gemini": "Gemini (Google)",
        "sarvam": "Sarvam AI (Native Indic)",
        "groq": "Llama 3.3 (Groq/Free)"
    }
    
    default_provider = os.getenv("GENERATOR_PROVIDER", "gemini").lower()
    default_label = provider_to_label.get(default_provider, "Gemini (Google)")
    options_list = list(provider_to_label.values())

    st.markdown("---")
    st.header("⚙️ Model Configuration")
    selected_model_ui = st.selectbox(
        "Select LLM Provider:",
        options=options_list,
        index=options_list.index(default_label),
        key="active_model_provider_selection_box"
    )
    
    model_provider_map = {
        "Gemini (Google)": "gemini",
        "Sarvam AI (Native Indic)": "sarvam",
        "Llama 3.3 (Groq/Free)": "groq"
    }
    active_provider = model_provider_map[selected_model_ui]

# Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about your uploaded document:"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        with st.spinner(f"Searching Qdrant and generating answer using {selected_model_ui}..."):
            try:
                # Run the entire query-retrieval-generation sequence inside a SINGLE loop call!
                full_response = run_async(run_query_pipeline(prompt, active_provider))
                
                message_placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")