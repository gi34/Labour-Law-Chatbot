import streamlit as st
import psycopg
from psycopg.rows import dict_row

from uuid import UUID, uuid4
from chatbot import build_vector_store, KB_PATH, get_database, rerank_documents, rag_chain, save_message


@st.cache_data(show_spinner=True)
def get_vector_store():
    vector_store = build_vector_store()
    return vector_store

vector_store = get_vector_store()


@st.cache_resource
def get_database_connection():
    connection = psycopg.connect(
        get_database(),
        autocommit=True,
        row_factory=dict_row,
    )

    with connection.cursor() as cursor:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_messages (
                id BIGSERIAL PRIMARY KEY,
                conversation_id UUID NOT NULL,
                role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                sources JSONB,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS chat_messages_conversation_idx
            ON chat_messages (conversation_id, created_at)
        """)
    return connection


connection = get_database_connection()

def render_chat_interface():
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = str(uuid4())

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
            background: #273449;
            padding: 16px;
            border-radius: 10px;
            font-size: 0.9rem;
            margin-top: 10px;
            color: #e2e8f0;
            border: 1px solid #475569;
    
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
            st.session_state.conversation_id = str(uuid4())
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
        save_message(connection, st.session_state.conversation_id, st.session_state.messages[-1])

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
            assistant_message = {
                "role": "assistant",
                "content": response,
                "sources": [doc.page_content for doc in reranked_docs]
            }
            st.session_state.messages.append(assistant_message)
            save_message(connection, st.session_state.conversation_id, assistant_message)

        except Exception as e:
            st.error(str(e))


if __name__ == "__main__":
    render_chat_interface()
