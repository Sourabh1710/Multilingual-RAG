import os
import sys
import types

# 1. Force single-threaded execution for PyTorch
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import streamlit as st

# 2. Sync Streamlit secrets into os.environ so all os.getenv() calls work throughout the pipeline modules
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
import os
import tempfile
from pathlib import Path
from dotenv import load_dotenv

# Import the modular pipeline classes
from rag_pipeline.ingestion.pipeline import IngestionPipeline
from rag_pipeline.embeddings.pipeline import EmbeddingPipeline
from rag_pipeline.retrieval.pipeline import RetrievalPipeline
from rag_pipeline.generation.pipeline import GenerationPipeline
from rag_pipeline.embeddings.model import EmbeddingModel

# Load environment variables
load_dotenv()

# Declare Global Constants
COLLECTION_NAME = "streamlit_user_manuals"

st.set_page_config(
    page_title="Indic Multilingual RAG",
    page_icon="🤖",
    layout="wide"
)

def run_async(coro):
    """
    Runs a coroutine in a fresh event loop each time to prevent loop-reuse hangs.
    """
    return asyncio.run(coro)


@st.cache_resource
def get_pipelines():
    try:
        shared_model = EmbeddingModel()
        ingest = IngestionPipeline(chunk_size=800, chunk_overlap=150)
        embed = EmbeddingPipeline(collection_name=COLLECTION_NAME, model=shared_model)
        retrieval = RetrievalPipeline(collection_name=COLLECTION_NAME, model=shared_model)
        generation = GenerationPipeline()
        return ingest, embed, retrieval, generation
    except Exception as e:
        st.error(f"Failed to initialize pipelines: {e}")
        st.stop()

ingest_pipeline, embed_pipeline, retrieval_pipeline, generation_pipeline = get_pipelines()

async def process_and_index_file(temp_file_path: Path) -> int:
    chunks = await ingest_pipeline.ingest_file(temp_file_path)
    if chunks:
        await embed_pipeline.embed_and_store(chunks)
        return len(chunks)
    return 0

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
                    # Clear the old database collection to prevent data pollution
                    run_async(embed_pipeline.store.clear_collection(COLLECTION_NAME))
                    
                    # Run the ingestion pipeline cleanly
                    total_chunks = run_async(process_and_index_file(temp_path))
                    
                    os.unlink(temp_path)
                    st.success(f"Success! Generated and indexed {total_chunks} chunks.")
                    st.session_state["ingested_file"] = uploaded_file.name
                except Exception as e:
                    st.error(f"Ingestion failed: {str(e)}")

    # DYNAMICALLY MAPPED MODEL CONFIGURATION
    provider_to_label = {
        "gemini": "Gemini (Google)",
        "sarvam": "Sarvam AI (Native Indic)",
        "groq": "Llama 3.3 (Groq/Free)"
    }
    
    # Automatically select the default option based on the active GENERATOR_PROVIDER secret
    default_provider = os.getenv("GENERATOR_PROVIDER", "gemini").lower()
    default_label = provider_to_label.get(default_provider, "Gemini (Google)")
    options_list = list(provider_to_label.values())

    st.markdown("---")
    st.header("⚙️ Model Configuration")
    selected_model_ui = st.selectbox(
        "Select LLM Provider:",
        options=options_list,
        index=options_list.index(default_label) # Matches the environment secret automatically
    )
    
    model_provider_map = {
        "Gemini (Google)": "gemini",
        "Sarvam AI (Native Indic)": "sarvam",
        "Llama 3.3 (Groq/Free)": "groq"
    }
    active_provider = model_provider_map[selected_model_ui]

    st.markdown("---")
    st.header("⚙️ Model Configuration")
    selected_model_ui = st.selectbox(
    "Select LLM Provider:",
    options=["Gemini (Google)", "Sarvam AI (Native Indic)", "Llama 3.3 (Groq/Free)"],
    index=0,
    key="llm_provider_select",
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
                # Retrieve matching chunks asynchronously
                chunks = run_async(retrieval_pipeline.retrieve(query=prompt, top_k=2))
                
                if chunks:
                    # Generate answer asynchronously
                    answer = run_async(
                        generation_pipeline.generate_answer(
                            query=prompt, 
                            context_chunks=chunks,
                            provider=active_provider
                        )
                    )
                    sources_text = "\n\n**Sources:**\n" + "\n".join([f"- Page {c.metadata.page_number} ({c.metadata.source_file})" for c in chunks])
                    full_response = f"{answer}{sources_text}"
                else:
                    full_response = "मुझे नहीं पता। (No matching documents found in database)."
                
                message_placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")