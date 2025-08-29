# app/chains.py
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from operator import itemgetter

from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama

from .config import settings, VECTOR_DIR

SYSTEM_PROMPT = """
You are a precise research assistant. Answer only from the provided context.
If the answer is not in the context, say you do not know.
Always include citations like [source p. N] if page is available.
""".strip()

ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "Question: {question}\n\nContext:\n{context}\n\nWrite a concise answer first. Then list bullet citations.")
])

def _format_docs(docs):
    parts = []
    for d in docs:
        src = d.metadata.get("source") or d.metadata.get("file_path") or d.metadata.get("url") or "unknown"
        page = d.metadata.get("page")
        head = f"[source={src} p. {page}]" if page is not None else f"[source={src}]"
        parts.append(f"{head}\n{d.page_content}")
    return "\n\n".join(parts)

def build_retriever(top_k: int = 5):
    embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)
    vs = Chroma(
        collection_name=settings.collection_name,
        embedding_function=embeddings,
        persist_directory=str(VECTOR_DIR),
    )
    return vs.as_retriever(search_kwargs={"k": top_k})

def build_qa_chain(model_name: str = "llama2-uncensored:7b"):
    llm = ChatOllama(model=model_name, temperature=0.2)
    retriever = build_retriever()

    rag = (
        RunnableParallel({
            # feed only the question string into retriever
            "context": itemgetter("question") | retriever | _format_docs,
            # also pass the raw question string to the prompt
            "question": itemgetter("question"),
        })
        | ANSWER_PROMPT
        | llm
        | StrOutputParser()
    )
    return rag, retriever