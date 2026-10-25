from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage, convert_to_messages
from reranking.reranker import rerank_documents
from retrieving_and_answering import semantic_retriever, rrf, bm25_retriever
from retrieving_and_answering.query_rewriting import rewrite_query
import config

load_dotenv(override=True)

MODEL = 'gemini-3.1-flash-lite'
DB_NAME = config.DB_NAME

llm = ChatGoogleGenerativeAI(
    model = MODEL,
    temperature = 0
)

SYSTEM_PROMPT = """
You are a knowledgeable, professional assistant representing the university NORTHBRIDGE INSTITUTE OF TECHNOLOGY.
You have to answer to student's queries.
Keep your language simple and easy to understand for students, but also it has to be professional as it's a university chatbot.
You have to answer based on the context provided.
If the user's query cannot be resolved by the context received, say so, no need of generating your own answers.

For follow-up questions, use the conversation to understand what the student means, but treat only the retrieved context as evidence. 
If the retrieved passages do not support an answer, say that you cannot verify it from the retrieved passages. 
Do not claim that the university documents lack the information, and do not contradict an earlier answer without supporting evidence.

If the student asks broadly what this assistant knows, covers, or can help with, describe only the areas in the Coverage List below. 
Do not infer the assistant's overall coverage from the retrieved passages.
Coverage List: academic regulations, attendance, examinations, courses, fees, scholarships, internships, placements, hostel, library, and student services.

Context:
{context} 

"""


def answer_question(question, history=None) -> tuple[str, list[Document]]:
    history = history or []
    search_query = rewrite_query(question, history)

    semantic_docs = semantic_retriever.fetch_documents(question)
    bm25_docs = bm25_retriever.keyword_retriever(question)

    result_lists = [bm25_docs, semantic_docs]
    if search_query != question.strip():
        result_lists.append(bm25_retriever.keyword_retriever(search_query)[:5])
        result_lists.append(semantic_retriever.fetch_documents(search_query)[:5])

    retrieved_docs = rrf.reciprocal_rank_fusion(result_lists)

    reranked_docs = rerank_documents(search_query, retrieved_docs)
    context = "\n\n".join(doc.page_content for doc in reranked_docs)
    sys_prompt = SYSTEM_PROMPT.format(context = context)
    messages = [SystemMessage(content=sys_prompt)]
    messages.extend(convert_to_messages(history))
    messages.append(HumanMessage(content = question))
    response = llm.invoke(messages)
    return response.content[0].get('text', ''), reranked_docs
