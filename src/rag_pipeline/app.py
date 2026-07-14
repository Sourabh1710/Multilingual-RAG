import os
import sys
import types

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import streamlit as st
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

# THE ASYNC RUNNER WRAPPER
def run_async(coro):
    """
    Safely runs an async coroutine inside Streamlit's synchronous environment,
    preserving persistent database and API connection pools without causing loop crashes.
    Fixed it after successful run once :)
    """
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(coro)


@st.cache_resource
def get_pipelines():
    # 1. Instantiate the heavy EmbeddingModel EXACTLY ONCE
    shared_model = EmbeddingModel()

    # 2. Pass the shared model as a reference to both pipelines!
    ingest = IngestionPipeline(chunk_size=800, chunk_overlap=150)
    embed = EmbeddingPipeline(collection_name=COLLECTION_NAME, model=shared_model)
    retrieval = RetrievalPipeline(collection_name=COLLECTION_NAME, model=shared_model)
    generation = GenerationPipeline()
    return ingest, embed, retrieval, generation

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

    st.markdown("---")
    st.header("⚙️ Model Configuration")
    selected_model_ui = st.selectbox(
        "Select LLM Provider:",
        options=["Gemini (Google)", "Sarvam AI (Native Indic)", "Llama 3.3 (Groq/Free)"],
        index=0
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