from pathlib import Path
from typing import List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from sentence_transformers import CrossEncoder
from operator import itemgetter
from psycopg.types.json import Jsonb
from uuid import UUID
from dotenv import load_dotenv
import os


load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
KB_PATH = Path(__file__).resolve().parent / "Labour Law.md"

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
    model="openai/gpt-oss-20b",
    api_key=GROQ_API_KEY,
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

# return database url
def get_database():
    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise ValueError(
            "DATABASE_URL is missing. Please add your Render PostgreSQL External Database URL to .env"
        )

    if "sslmode=" not in database_url:
        separator = "&" if "?" in database_url else "?"
        database_url = f"{database_url}{separator}sslmode=require"

    return database_url

def save_message(connection, conversation_id: str, message: dict) -> None:
    with connection.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO chat_messages (conversation_id, role, content, sources)
            VALUES (%s, %s, %s, %s)
            """,
            (
                UUID(conversation_id),
                message["role"],
                message["content"],
                Jsonb(message.get("sources")) if message.get("sources") else None,
            ),
        )
