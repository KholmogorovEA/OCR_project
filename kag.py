from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv
load_dotenv()
from app.config import settings

os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"


urls = [
    "https://www.consultant.ru/document/cons_doc_LAW_5142/c4fe6e6c3382269311df4bffaf438feb330600cf/?utm_source=chatgpt.com",
    "https://www.consultant.ru/document/cons_doc_LAW_39570/6b58bd70c3176240c638ba06b5852fb675edd935/?utm_source=chatgpt.com",
    "https://www.consultant.ru/document/cons_doc_LAW_39570/76602a8eaa07ea6be840dcada31c74e26f1c0600/?utm_source=chatgpt.com",
    "https://www.consultant.ru/document/cons_doc_LAW_39570/39601726e6ae21d94355755b80bff29f8e9b28d7/?utm_source=chatgpt.com",
    "https://www.consultant.ru/document/cons_doc_LAW_39570/39359d15e88b9690c68d9a8de1f922c10e64a248/",
    "https://www.consultant.ru/document/cons_doc_LAW_39570/76602a8eaa07ea6be840dcada31c74e26f1c0600/",
    "https://www.consultant.ru/document/cons_doc_LAW_286833/e52b8c67d750ba20306e29c4d004c58f224e870f/",
]

docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0
)
doc_splits = text_splitter.split_documents(docs_list)

# Add to vectorDB
vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chroma",
    embedding=OpenAIEmbeddings(model="text-embedding-3-large"),
)
retriever = vectorstore.as_retriever(search_type="mmr", k=10, search_kwargs={'k': 10, 'lambda_mult': 0.50})

### Retrieval Grader

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from pydantic import BaseModel, Field


# Data model
class GradeDocuments(BaseModel):


    binary_score: str = Field(
        description="Документы релевантны вопросу, 'Да' или 'Нет'"
    )


# LLM with function call
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
structured_llm_grader = llm.with_structured_output(GradeDocuments)

# Prompt
system = """YВы — эксперт, оценивающий релевантность извлечённого документа к пользовательскому вопросу.
Если документ содержит ключевые слова или семантическое значение, связанные с вопросом, оцените его как релевантный.
Дайте бинарную оценку "yes" или "no", чтобы указать, является ли документ релевантным вопросу."""
grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Извлечённый документ: \n\n {document} \n\n Вопрос юзера: {question}"),
    ]
)

retrieval_grader = grade_prompt | structured_llm_grader
question = """Найти юридические аргументы и правовые ссылки для защиты Ответчика, включая постановления, решения судов, законодательные акты, регулирующие 
[ключевые аспекты, например, ответственность сторон, обязательства по договору и т.д.]. Учитывая позиции судебной практики и законодательство, сформулируй линию защиты"""
docs = retriever.invoke(question)
doc_txt = docs[1].page_content
print(retrieval_grader.invoke({"question": question, "document": doc_txt}))

### Generate

from langchain import hub
from langchain_core.output_parsers import StrOutputParser

# Prompt
prompt = hub.pull("rlm/rag-prompt")

# LLM
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0) # type: ignore


# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# Chain
rag_chain = prompt | llm | StrOutputParser()

# Run
generation = rag_chain.invoke({"context": docs, "question": question})
print(generation)

### Question Re-writer

# LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_completion_tokens=2500)

# Prompt
system = """Вы — переформулировщик вопросов, который преобразует входной вопрос в улучшенную версию, 
оптимизированную для веб-поиска. Проанализируйте ввод и постарайтесь выявить его основное смысловое намерение и значение."""
re_write_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            "human",
            "Вот исходный вопрос: \n\n {question} \n Сформулируйте улучшенный вариант этого вопроса.",
        ),
    ]
)

question_rewriter = re_write_prompt | llm | StrOutputParser()
question_rewriter.invoke({"question": question})

### Search
# os.environ["TAVILY_API_KEY"] = "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"
os.environ["TAVILY_API_KEY"] = settings.TAVILY_KEY
from langchain_community.tools.tavily_search import TavilySearchResults

web_search_tool = TavilySearchResults(api_key=os.getenv("TAVILY_API_KEY"), k=5) # type: ignore

from typing import List

from typing_extensions import TypedDict


class GraphState(TypedDict):
    """
    Представляет состояние нашего графа.

    Атрибуты:

        question: вопрос
        generation: генерация LLM
        web_search: необходимость добавления поиска
        documents: список документов
"""

    question: str
    generation: str
    web_search: str
    documents: List[str]

from langchain.schema import Document


def retrieve(state):

    print("---RETRIEVE---")
    question = state["question"]

    # Retrieval
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}


def generate(state):

    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]

    # RAG generation
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}


def grade_documents(state):


    print("---ПРОВЕРКА РЕЛЕВАНТНЫХ ДОКУМЕНТОВ---")
    question = state["question"]
    documents = state["documents"]

    # Score each doc
    filtered_docs = []
    web_search = "No"
    for d in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        grade = score.binary_score # type: ignore
        if grade == "yes":
            print("---GRADE: РЕЛЕВАНТНЫЙ ДОКУМЕНТ---")
            filtered_docs.append(d)
        else:
            print("---GRADE: ДОКУМЕНТ РЕЛЕВАНТНЫЙ---")
            web_search = "Yes"
            continue
    return {"documents": filtered_docs, "question": question, "web_search": web_search}


def transform_query(state):


    print("---ТРАНСФОРМИРОВАННЫЙ ЗАПРОС---")
    question = state["question"]
    documents = state["documents"]

    # Re-write question
    better_question = question_rewriter.invoke({"question": question})
    return {"documents": documents, "question": better_question}


def web_search(state):


    print("---WEB SEARCH---")
    question = state["question"]
    documents = state["documents"]

    # Web search
    docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)
    documents.append(web_results)

    return {"documents": documents, "question": question}


### Edges


def decide_to_generate(state):


    print("---ASSESS GRADED DOCUMENTS---")
    state["question"]
    web_search = state["web_search"]
    state["documents"]

    if web_search == "Yes":
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print(
            "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---"
        )
        return "transform_query"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"

from langgraph.graph import END, StateGraph, START

workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("retrieve", retrieve)  # retrieve
workflow.add_node("grade_documents", grade_documents)  # grade documents
workflow.add_node("generate", generate)  # generatae
workflow.add_node("transform_query", transform_query)  # transform_query
workflow.add_node("web_search_node", web_search)  # web search

# Build graph
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "transform_query": "transform_query",
        "generate": "generate",
    },
)
workflow.add_edge("transform_query", "web_search_node")
workflow.add_edge("web_search_node", "generate")
workflow.add_edge("generate", END)

# Compile
app = workflow.compile()

from pprint import pprint

# Run
inputs = {"question": "Надйди и предоставь аргументы описывающие линию защиты Ответчика, приведи примеры, ссылки на статьи, контекст и описание"}
for output in app.stream(inputs):
    for key, value in output.items():
        # Node
        pprint(f"Node '{key}':")
        # Optional: print full state at each node
        # pprint.pprint(value["keys"], indent=2, width=80, depth=None)
    pprint("\n---\n")

# Final generation
pprint(value["generation"]) # type: ignore