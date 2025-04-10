


# """
# Полная система юридического поиска с мульти-ретрейверами и улучшенной точностью
# """
# import sys
# from pathlib import Path

# # Добавляем корневую папку проекта в sys.path
# # sys.path.append(str(Path(__file__).parent.parent.parent))
# # ██ Импорты
# # ██ Импорты
# import asyncio
# import numpy as np
# import faiss
# from typing import List, Dict
# from pydantic import BaseModel, Field
# from langchain_openai import ChatOpenAI  # Изменено!
# from langchain.retrievers import EnsembleRetriever  # Изменено!
# from langchain_core.prompts import ChatPromptTemplate, PromptTemplate  # Изменено!
# from langchain_core.documents import Document  # Изменено!
# from langchain_community.cache import SQLiteCache  # Изменено!
# from transformers import pipeline
# from dotenv import load_dotenv
# load_dotenv()
# from app.config import settings
# #from app.main import index_deep_pavlov_dialog_invest, index_deep_pavlov_sedoy, index_deep_pavlov_zakaryan_ashot, index_deep_pavlov_zakaryan_susanna, index_deep_pavlov_processed_text # index_sentence_transformers_dialog_invest, index_sentence_transformers_processed_text, index_sentence_transformers_sedoy, index_sentence_transformers_zakaryan_ashot, index_sentence_transformers_zakaryan_susanna

# index_deep_pavlov_processed_text = faiss.read_index("C:\\Users\\Evgenii\\Desktop\\OCR_bot\\transformersDeepPavlov\\processed_text_faiss_index.index") # type: ignore
# index_deep_pavlov_dialog_invest = faiss.read_index("C:\\Users\\Evgenii\\Desktop\\OCR_bot\\transformersDeepPavlov\\dialog_invest_faiss_index.index") # type: ignore
# index_deep_pavlov_sedoy = faiss.read_index("C:\\Users\\Evgenii\\Desktop\\OCR_bot\\transformersDeepPavlov\\sedoy_faiss_index.index") # type: ignore
# index_deep_pavlov_zakaryan_ashot = faiss.read_index("C:\\Users\\Evgenii\\Desktop\\OCR_bot\\transformersDeepPavlov\\zakaryan_ashot_faiss_index.index") # type: ignore
# index_deep_pavlov_zakaryan_susanna = faiss.read_index("C:\\Users\\Evgenii\\Desktop\\OCR_bot\\transformersDeepPavlov\\zakaryan_susanna_faiss_index.index") # type: ignore

# # ██ Инициализация моделей и кеша
# llm = ChatOpenAI(
#     temperature=0,
#     cache=SQLiteCache("legal_cache.db")  # Кеширование результатов
# )

# # ██ Модели данных
# class LegalRelevance(BaseModel):
#     relevance: float = Field(..., ge=0, le=1, description="Оценка релевантности 0-1")
#     jurisdiction: str = Field(..., description="Применимая юрисдикция")

# class RetrieverInfo(BaseModel):
#     name: str
#     description: str
#     examples: List[str]
#     retriever: None # type: ignore

# # ██ Конфигурация ретрейверов
# retriever_infos = [
#     RetrieverInfo(
#         name="Банкротсво Диалог Инвест",
#         description="""Suitable for answering questions about Application of ZELO LLC for inclusion in the register of claims of Dialog Invest with an inventory and a check. Application of OOO ZELO on reduction of the amount of claims against the debtor on 3l. Debt calculation from the Tverskoy Court case. Notice of the meeting of creditors of Dialog Invest on 22.08.2023. 2024-10-14 Notice of the meeting of creditors of Dialog Invest (inc.18.10.24) State duty on appeal Zelo. A32-28626-2021_20221024_Reshenija_i_postanovlenija Признание банкротом. A32-28626-2021_20230619_Opredelenie On the Election of the Arbitration Administrator. A32-28626-2021_20231113_Opredelenie Inclusion of Zelo Claims. A32-28626-2021_20240603_Opredelenie Postponement of the procedure. A32-28626-2021_20240816_Opredelenie Remanding the succession to Vladimirska without motion. A32-28626-2021_20240816_Opredelenie Termination of the procedure. A32-28626-2021_20241121_Postanovlenie_apelljacionnoj_instancii Отмена прекращения.""",
#         examples=["Заявление ООО ЗЕЛО о включении в реестр требований Диалог Инвест с описью и чеком", "Заявление ООО ЗЕЛО об уменьшении суммы требований к должнику на 3л", "Уведомление о собрании кредиторов Диалог Инвест", "A32-28626-2021_20221024_Reshenija_i_postanovlenija Признание банкротом", "A32-28626-2021_20240816_Opredelenie Прекращение процедуры"],
#         retriever=index_deep_pavlov_dialog_invest  # Заменить на реальный ретрейвер # type: ignore
#     ),
#     RetrieverInfo(
#         name="Банкротсво Седого",
#         description="""Suitable for answering questions about Appeal in case 2-523-2022 from Sedoy A.A. (inh.19.03.24). 2024-04-15 Appeal. complaint from Sedoy A.A. case A32-48721_2023 (inh.26.04.24) 2024-04-15 Appeal. complaint from Sedoy A.A. case A32-48721_2023 (inh.26.04.24)-1. Notification about the meeting of creditors by Sedoy A.A. (inh.21.05.24). A32-48721-2023_20231009_Opredelenie Acceptance of the application. A32-48721-2023_20231206_Opredelenie. A32-48721-2023_20240227_Opredelenie Введение реструктуризации. A32-48721-2023_20240517_Opredelenie. A32-48721-2023_20240909_Opredelenie_Протокол. Sedoy is address #1. Sedoy A.A. to address #2.""",
#         examples=["Апелляционная жалоба по делу 2-523-2022 от Седой А.А. (вх.19.03.24)", "Уведомление о собрании кредиторов Седой А.А. (вх.21.05.24)", "A32-48721-2023_20240227_Opredelenie Введение реструктуризации", "Седой - адрес №1. Седой - адрес №2."],
#         retriever=index_deep_pavlov_sedoy # type: ignore
#     ),
#     RetrieverInfo(
#         name="Банкротство Закарян Ашота",
#         description="""Suitable for answering questions about ZELO LLC bankruptcy application of ZAKARYAN A.A. Все про Закаряна Ашота. with an inventory and a check. Notice of the meeting of creditors Zakaryan A.A. (vh.21.05.24) A41-35102-2023_20230607_Opredelenie. A41-35102-2023_20241107_Opredelenie Replacement of Zelo by Vladimirskaya in part of the claims. Zorya to Zakarian for 26 mln. Kozeruk to Zakarian under the receipt. Decision on the claim of MC Veles Management to Zakaryans. A41-35102-2023_20231207_Opredelenie On reclamation of evidence. A41-35102-2023_20231207_Opredelenie On refusal to demand evidence from the Civil Registry Office. A41-35102-2023_20240326_Opredelenie On transfer of documents. A41-35102-2023_20241010_Opredelenie Assignment of claims to Artel by Subsidiary""",
#         examples=["очередь наследства", "завещание"],
#         retriever=index_deep_pavlov_zakaryan_ashot # type: ignore
#     ),
#     RetrieverInfo(
#         name="Банкротство Закарян Сюзанна",
#         description="Suitable for answering questions about Suzanne Zakarian and all information related to her. Kopija dogovora zajma 27012020015-MSB ot 27 janvarja 2020 goda. 4.Kopija dogovora poruchitelstva ot 27 janvarja 2020 goda. Kopija dogovoru ustupki 04-03-22 BD-Faktorius ot 04 marta 2022 goda. AO_PKO_YUB_FAKTORIUS. ZELO bankruptcy petition of Zakaryan Susanna with inventory and check. Opredelenie Acceptance of the application.Opredelenie Acceptance of the application of Zelo (Susanna). A41-30605-2023_20230706_Opredelenie. Leaving the Factorius application without consideration. Reshenija_i_postanovlenija Recognition of Susanna as bankrupt. Claims of SME Bank. Opredelenie Claims of Artel without motion.",
#         examples=["Закарян Сюзанна", "Заявление ЗЕЛО о банкротстве Закарян Сусанны с описью и чеком", "ставление заявления Факториус без рассмотрения", "Reshenija_i_postanovlenija Признание Сусанны банкротом"],
#         retriever=index_deep_pavlov_zakaryan_susanna # type: ignore
#     ),
#     RetrieverInfo(
#         name="Банкротство Артель",
#         description="Suitable for answering questions about bankruptcy ARTEL. ARTEL - APPLICATION (for an injunction against a PSC). Power of attorney notarized by Artel for Zakaryan Ashot. Statement of accounts of Artel. Power of Attorney Artel for Altshuler. Analysis of financial condition of LLC ARTEL. Conclusion on transactions Finanalysis of LLCARTEL. answers of reg bodies of the Ministry of Internal Affairs (1). answers of reg bodies. Report on the activities of the temporary manager in the AC dated 20.07.2022. Submission of the candidacy of the insolvency administrator 1l. Minutes of the meeting of creditors 2. Register of creditors as of 29.07.2022. Claims behind the register of creditors_ as of 29.07.2022. Notifications and requests.",
#         examples=["банкротство Артель", "собрание кредиторов", "Доверенность нотариальная Артель на Закаряня Ашота", "редставление кандидатуры арбитражного управляющего 1л. Протокол собрания кредиторов 2. Реестр кредиторов на 29.07.2022. Требования за реестром кредиторов_на 29.07.2022. Уведомления и запросы"],
#         retriever=index_deep_pavlov_processed_text # type: ignore
#     ),
#     RetrieverInfo(
#         name="Наследственное право",
#         description="Оформление наследства, споры между наследниками",
#         examples=["очередь наследства", "завещание"],
#         retriever=index_sentence_transformers_dialog_invest # type: ignore
#     ),
#     RetrieverInfo(
#         name="Наследственное право",
#         description="Оформление наследства, споры между наследниками",
#         examples=["очередь наследства", "завещание"],
#         retriever=index_sentence_transformers_processed_text # type: ignore
#     ),
#     RetrieverInfo(
#         name="Наследственное право",
#         description="Оформление наследства, споры между наследниками",
#         examples=["очередь наследства", "завещание"],
#         retriever=index_sentence_transformers_sedoy # type: ignore
#     ),
#     RetrieverInfo(
#         name="Наследственное право",
#         description="Оформление наследства, споры между наследниками",
#         examples=["очередь наследства", "завещание"],
#         retriever=index_sentence_transformers_zakaryan_ashot # type: ignore
#     ),
#     RetrieverInfo(
#         name="Наследственное право",
#         description="Оформление наследства, споры между наследниками",
#         examples=["очередь наследства", "завещание"],
#         retriever=index_sentence_transformers_zakaryan_susanna # type: ignore
#     )
# ]

# # ██ Классификатор с контекстными примерами
# classifier_prompt = PromptTemplate.from_template(
#     """
# Определи юридическую категорию вопроса. Категории:
# {category_descriptions}

# Примеры классификации:
# - "Как взыскать убытки?" → "Договорное право"
# - "Срок принятия наследства" → "Наследственное право"

# Вопрос: {query}
# Категория:"""
# )

# async def classify_query(query: str) -> Dict:
#     """Классификация с динамическим порогом уверенности"""
#     # Формируем описания категорий
#     category_descriptions = "\n".join([
#         f"{info.name}: {info.description} (Примеры: {', '.join(info.examples)})"
#         for info in retriever_infos
#     ])
    
#     # Классификация через LLM
#     response = await llm.ainvoke(
#         classifier_prompt.format(
#             category_descriptions=category_descriptions,
#             query=query
#         )
#     )
    
#     # Получаем оценки для всех категорий
#     scores = await asyncio.gather(*[
#         llm.ainvoke(f"Оцени релевантность категории '{info.name}' для вопроса: {query} (0-1)")
#         for info in retriever_infos
#     ])
    
#     scores = [float(s.content) for s in scores] # type: ignore
#     best_idx = np.argmax(scores)
    
#     return {
#         "labels": [info.name for info in retriever_infos],
#         "scores": scores,
#         "best_category": retriever_infos[best_idx].name,
#         "best_score": scores[best_idx]
#     }

# # ██ Динамический порог уверенности
# confidence_scores = []

# def calculate_dynamic_threshold(quantile=0.85):
#     return np.quantile(confidence_scores, quantile) if confidence_scores else 0.7

# # ██ Ансамбль ретрейверов с весами
# def create_ensemble(retrievers: List, scores: List[float]):
#     # Нормализация и квадратичное усиление
#     weights = np.array(scores)**2
#     weights /= weights.sum()
    
#     return EnsembleRetriever(
#         retrievers=retrievers,
#         weights=weights.tolist()
#     )

# # ██ Reciprocal Rank Fusion (RRF)
# def reciprocal_rank_fusion(results: List[List[Document]], k=60):
#     fused_scores = {}
#     for docs in results:
#         for rank, doc in enumerate(docs):
#             doc_str = doc.page_content
#             if doc_str not in fused_scores:
#                 fused_scores[doc_str] = 0
#             fused_scores[doc_str] += 1/(rank + k)
    
#     return sorted(
#         [(doc, score) for doc, score in fused_scores.items()],
#         key=lambda x: x[1], 
#         reverse=True
#     )

# # ██ Юридический градер
# legal_grader_prompt = ChatPromptTemplate.from_messages([
#     ("system", "Ты эксперт-юрист. Оцени:"),
#     ("human", "Док: {doc}\nВопрос: {query}\nОценка релевантности (0-1) и юрисдикция:")
# ])

# async def grade_document(doc: Document, query: str) -> LegalRelevance:
#     """Оценка релевантности документа"""
#     response = await llm.ainvoke(
#         legal_grader_prompt.format(
#             doc=doc.page_content[:2000],  # Обрезаем длинные документы
#             query=query
#         )
#     )
#     return LegalRelevance.parse_raw(response.content) # type: ignore

# # ██ Основной поток обработки
# async def legal_rag_system(query: str):
#     """Полный цикл обработки запроса"""
    
#     # 1. Классификация запроса
#     classification = await classify_query(query)
#     confidence_scores.append(classification["best_score"])
#     threshold = calculate_dynamic_threshold()
    
#     # 2. Выбор стратегии поиска
#     if classification["best_score"] >= threshold:
#         # Выбор одного ретрейвера
#         retriever = next(
#             info.retriever for info in retriever_infos 
#             if info.name == classification["best_category"]
#         )
#         results = await retriever.aget_relevant_documents(query, limit=10) # type: ignore
#     else:
#         # Ансамблевый поиск
#         retrievers = [info.retriever for info in retriever_infos]
#         ensemble = create_ensemble(retrievers, classification["scores"])
#         results = await ensemble.aget_relevant_documents(query)
    
#     # 3. Ранжирование и фильтрация
#     graded_results = await asyncio.gather(*[
#         grade_document(doc, query) for doc in results
#     ])
    
#     filtered_docs = [
#         doc for doc, grade in zip(results, graded_results)
#         if grade.relevance >= 0.65
#     ]
    
#     # 4. Генерация ответа
#     context = "\n\n".join([d.page_content for d in filtered_docs[:5]])
#     response = await llm.ainvoke(
#         f"Ответь на вопрос основываясь на документах:\n{context}\n\nВопрос: {query}"
#     )
    
#     return {
#         "answer": response.content,
#         "used_sources": [d.metadata.get("source") for d in filtered_docs[:3]],
#         "confidence": classification["best_score"],
#         "jurisdiction": graded_results[0].jurisdiction if filtered_docs else "N/A"
#     }

# # ██ Пример использования
# if __name__ == "__main__":
#     query = "Какие сроки для оспаривания завещания?"
    
#     result = asyncio.run(legal_rag_system(query))
    
#     print(f"Вопрос: {query}")
#     print(f"Ответ: {result['answer']}")
#     print(f"Источники: {', '.join(result['used_sources'])}")
#     print(f"Юрисдикция: {result['jurisdiction']}")
#     print(f"Уверенность классификации: {result['confidence']:.2f}")