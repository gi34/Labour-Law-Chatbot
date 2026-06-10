from pathlib import Path
from typing import List

import streamlit as st
from groq import Groq
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from sentence_transformers import CrossEncoder

KB_PATH = Path(__file__).resolve().parent / "Labour Law.txt"


def chunking(text: str) -> List[Document]:
    """Split text into a parent/child hierarchy for better retrieval."""
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=400,
        separators=["\n\n", "\n", " ", ""],
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""],
    )

    parent_chunks = parent_splitter.split_text(text)
    documents: List[Document] = []
    for parent_index, parent_text in enumerate(parent_chunks):
        child_chunks = child_splitter.split_text(parent_text)
        for child_index, child_text in enumerate(child_chunks):
            documents.append(
                Document(
                    page_content=child_text,
                    metadata={
                        "source": KB_PATH.name,
                        "parent_chunk": f"parent_{parent_index}",
                        "child_index": child_index,
                    },
                )
            )
    return documents


@st.cache_data(show_spinner=True)
def build_vector_store() -> FAISS:
    if not KB_PATH.exists():
        raise FileNotFoundError(f"Knowledge base not found at {KB_PATH}")

    loader = TextLoader(str(KB_PATH), encoding="utf-8")
    docs = loader.load()
    split_docs: List[Document] = []
    for doc in docs:
        split_docs.extend(chunking(doc.page_content))

    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(split_docs, embeddings)
    return vector_store


@st.cache_resource(show_spinner=True)
def get_cross_encoder() -> CrossEncoder:
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def rerank_documents(query: str, docs: List[Document], top_k: int = 5) -> List[Document]:
    if not docs:
        return []
    pairs = [(query, doc.page_content) for doc in docs]
    scores = get_cross_encoder().predict(pairs)
    ranked = sorted(zip(scores, docs), key=lambda item: item[0], reverse=True)
    return [doc for _, doc in ranked][:top_k]


def answer_query(query: str, vector_store: FAISS) -> str:
    retrieved_docs = vector_store.similarity_search(query, k=10)
    reranked_docs = rerank_documents(query, retrieved_docs, top_k=5)

    context = "\n\n".join(
        f"Source: {doc.metadata.get('source', 'Labour Law')}\n{doc.page_content}"
        for doc in reranked_docs
    )

    prompt = (
        "You are a Malaysian labour law assistant. Answer the question using only the provided context from the Akta Kerja 1955 knowledge base. "
        "If the answer is not contained in the context, say you cannot find a specific answer in the law text.\n\n"
        "Context:\n" + context + "\n\nQuestion: " + query + "\nAnswer:"
    )

    client = Groq(api_key="YOUR_GROQ_API_KEY")
    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        temperature=0.2,
        max_completion_tokens=1024,
        top_p=1,
        stream=True,
        stop=None,
    )

    output = []
    for chunk in completion:
        output.append(chunk.choices[0].delta.content or "")

    return "".join(output)


def render_chat_interface():
    st.set_page_config(
        page_title="Labour Law Assistant",
        page_icon="⚖️",
        layout="wide",
    )

    # ---------- CUSTOM CSS ----------
    st.markdown("""
    <style>
        .main {
            background-color: #f8fafc;
        }

        .stChatMessage {
            padding: 1rem;
            border-radius: 16px;
            margin-bottom: 1rem;
        }

        .stChatMessage[data-testid="chatAvatarIcon-user"] {
            background-color: #2563eb;
        }

        .stChatMessage[data-testid="chatAvatarIcon-assistant"] {
            background-color: #0f172a;
        }

        .block-container {
            padding-top: 2rem;
            max-width: 1100px;
        }

        .title {
            font-size: 2.5rem;
            font-weight: 700;
            color: #0f172a;
            margin-bottom: 0;
        }

        .subtitle {
            color: #475569;
            margin-top: 0;
            margin-bottom: 2rem;
        }

        .source-box {
            background: #e2e8f0;
            padding: 12px;
            border-radius: 10px;
            font-size: 0.9rem;
            margin-top: 10px;
        }
    </style>
    """, unsafe_allow_html=True)

    # ---------- SIDEBAR ----------
    with st.sidebar:
        st.title("⚖️ Labour Law")

        st.markdown("---")

        st.subheader("Knowledge Base")
        st.write(f"File: `{KB_PATH.name}`")

        st.subheader("Retrieval")
        top_k = st.slider("Retrieved Chunks", 3, 15, 10)
        rerank_k = st.slider("Reranked Chunks", 1, 10, 5)

        show_sources = st.toggle("Show Sources", value=True)

        st.markdown("---")

        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()

    # ---------- HEADER ----------
    st.markdown(
        "<p class='title'>Labour Law Assistant</p>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<p class='subtitle'>Ask questions about Malaysia's Akta Kerja 1955.</p>",
        unsafe_allow_html=True,
    )

    # ---------- SESSION ----------
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # ---------- CHAT HISTORY ----------
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message["role"] == "assistant" and message.get("sources"):
                if show_sources:
                    with st.expander("Sources Used"):
                        for src in message["sources"]:
                            st.markdown(
                                f"""
                                <div class="source-box">
                                    {src[:500]}...
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

    # ---------- USER INPUT ----------
    prompt = st.chat_input("Ask your labour law question...")

    if prompt:
        # Show user message
        st.chat_message("user").markdown(prompt)

        # Save user message
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })

        try:
            vector_store = build_vector_store()

            retrieved_docs = vector_store.similarity_search(prompt, k=top_k)
            reranked_docs = rerank_documents(
                prompt,
                retrieved_docs,
                top_k=rerank_k
            )

            context = "\n\n".join(
                doc.page_content for doc in reranked_docs
            )

            prompt_template = f"""
You are a Malaysian labour law assistant.

Answer ONLY using the provided context.

If the answer is not found, say:
"I cannot find a specific answer in the provided law text."

Context:
{context}

Question:
{prompt}

Answer:
"""

            client = Groq(api_key="YOUR_API_KEY")

            # ---------- STREAM RESPONSE ----------
            with st.chat_message("assistant"):
                response_placeholder = st.empty()

                streamed_text = ""

                completion = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[
                        {
                            "role": "user",
                            "content": prompt_template,
                        }
                    ],
                    temperature=0.2,
                    stream=True,
                )

                for chunk in completion:
                    token = chunk.choices[0].delta.content or ""
                    streamed_text += token

                    response_placeholder.markdown(streamed_text)

                # Sources
                if show_sources:
                    with st.expander("Sources Used"):
                        for doc in reranked_docs:
                            st.markdown(
                                f"""
                                <div class="source-box">
                                    {doc.page_content[:500]}...
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

            # Save assistant response
            st.session_state.messages.append({
                "role": "assistant",
                "content": streamed_text,
                "sources": [doc.page_content for doc in reranked_docs]
            })

        except Exception as e:
            st.error(str(e))


if __name__ == "__main__":
    render_chat_interface()
