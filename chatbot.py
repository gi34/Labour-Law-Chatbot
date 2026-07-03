from pathlib import Path
from typing import List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
from groq import Groq
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from sentence_transformers import CrossEncoder
from operator import itemgetter

KB_PATH = Path(__file__).resolve().parent / "Labour Law.md"
history = []  # Initialize history as an empty list

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

def rerank_documents(query: str, docs: List[Document], top_k: int = 5) -> List[Document]:
    if not docs:
        return []
    pairs = [(query, doc.page_content) for doc in docs]
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, docs), key=lambda item: item[0], reverse=True)
    return [doc for _, doc in ranked][:top_k]

vector_store = build_vector_store()

prompt = ChatPromptTemplate.from_messages([
    ("system",
    '''
    You are a Malaysian labour law assistant. Answer the question using only the provided context.
    If the question contains ambiguous or unclear terms, check the history first. 
    If the question remain ambiguous after checking the history, ask for clarification before answering.
    
    If the answer is not contained in the context, say you cannot find a specific answer in the law text.
    
    Context:
    {context}

    History:
    {history}

    Answer:
    '''
    ),(
    "human",
    "{question}")
])

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key="your key",
    temperature=0.2,
    max_completion_tokens=1024,
    top_p=1,
    stream=False,
    stop=None,
)


rag_chain = (
    {
        "context": itemgetter("context"),
        "question": itemgetter("question"),
        "history": itemgetter("history"),
    }
    | prompt
    | llm
    | StrOutputParser()
)


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
    user_prompt = st.chat_input("Ask your labour law question...")

    if user_prompt:
        with st.chat_message("user"):
            st.markdown(user_prompt)

        st.session_state.messages.append({
            "role": "user",
            "content": user_prompt
        })

        history = st.session_state.messages[:-3]  # Exclude the current user message

        history_text = ""
        if history:
            history_text = "\n\n".join(
                f"{m.get('role','')}: {m.get('content','')}" for m in history
            )

        try:
            retrieved_docs = vector_store.similarity_search(user_prompt, k=top_k)
            reranked_docs = rerank_documents(user_prompt, retrieved_docs, top_k=rerank_k)

            context = "\n\n".join(
                f"Source: {doc.metadata.get('source', 'Labour Law')}\n{doc.page_content}"
                for doc in reranked_docs
            )

            with st.chat_message("assistant"):
                response_placeholder = st.empty()

                response = ""
                
                for chunk in rag_chain.stream({
                    "context": context,
                    "question": user_prompt,
                    "history": history_text,
                }):
                    response += chunk
                    response_placeholder.markdown(response)

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
                "content": response,
                "sources": [doc.page_content for doc in reranked_docs]
            })

        except Exception as e:
            st.error(str(e))


if __name__ == "__main__":
    render_chat_interface()
