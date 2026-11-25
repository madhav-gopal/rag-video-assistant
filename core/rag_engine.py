import os

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from core.vector_store import build_vector_store, load_vector_store, get_retriever
from core.llm_utils import get_llm, invoke_with_retry

_RAG_SYSTEM_PROMPT = """\
You are an expert meeting assistant. Answer the user's question
based ONLY on the meeting transcript context provided below.

If the answer is not found in the context, say:
"I could not find this information in the meeting transcript."

Always be concise and precise. If quoting someone, mention it clearly.

Context from meeting transcript:
{context}"""


def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])


def _build_chain(retriever):
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", _RAG_SYSTEM_PROMPT),
        ("human", "{question}"),
    ])
    return (
        {
            "context":  retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )


def build_rag_chain(transcript: str):
    vector_store = build_vector_store(transcript)
    retriever    = get_retriever(vector_store, k=4)
    return _build_chain(retriever)


def load_rag_chain():
    vector_store = load_vector_store()
    retriever    = get_retriever(vector_store)
    return _build_chain(retriever)


def ask_question(rag_chain, question: str) -> str:
    print(f"Question: {question}")
    # ✅ Retry wrapper handles 429 automatically
    return invoke_with_retry(rag_chain, question, label="rag-question")
