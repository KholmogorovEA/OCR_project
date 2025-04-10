import os
import hashlib
import uuid
import numpy as np
import faiss
import pickle
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownTextSplitter
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer


class MarkdownProcessor:
    def __init__(self):
        # Инициализация моделей
        self.splitter = MarkdownTextSplitter(chunk_size=2500, chunk_overlap=250)
        self.kw_model = KeyBERT()

    def process_markdown_files(self, folder_path: str) -> List[Document]:
        """Обработка всех Markdown-файлов в папке"""
        all_docs = []

        for filename in os.listdir(folder_path):
            if filename.endswith(".md"):
                file_path = os.path.join(folder_path, filename)
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                # Генерация уникального хеша файла
                file_hash = hashlib.md5(content.encode()).hexdigest()

                # Разбиение на чанки
                chunks = self.splitter.split_text(content)

                # Обработка чанков
                for chunk in chunks:
                    metadata = self.generate_metadata(chunk, file_hash)
                    doc = Document(
                        page_content=chunk,
                        metadata=metadata
                    )
                    all_docs.append(doc)

        return all_docs

    def generate_metadata(self, text: str, file_hash: str) -> dict:
        """Генерация метаданных для чанка"""
        metadata = {}

        # 1. Извлечение заголовков
        metadata["headers"] = self.extract_headers(text)

        # 2. Извлечение ключевых слов с KeyBERT
        metadata["keywords"] = self.extract_keywords(text)

        # 3. Технические метаданные
        metadata.update({
            "file_hash": file_hash,
            "text_length": len(text),
            "chunk_id": str(uuid.uuid4())  # Уникальный идентификатор для чанка
        })

        return metadata

    def extract_headers(self, text: str) -> List[str]:
        """Извлечение заголовков Markdown"""
        headers = []
        for line in text.split("\n"):
            if line.startswith("#"):
                header = line.strip("# ").strip()
                headers.append(header)
        return headers

    def extract_keywords(self, text: str, top_n: int = 33) -> List[str]:
        """Извлечение ключевых слов с KeyBERT"""
        keywords = self.kw_model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 2),
            stop_words=None, # type: ignore
            top_n=top_n,
            use_mmr=True
        )
        return [kw[0] for kw in keywords]  # type: ignore


class VectorIndexer:
    def __init__(self, model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"):
        self.index = None
        self.doc_store = {}
        self.model = SentenceTransformer(model_name)

    def create_embeddings(self, docs: List[Document], batch_size: int = 32) -> np.ndarray:
        """Создание эмбеддингов с SentenceTransformers"""
        texts = [doc.page_content for doc in docs]
        embeddings = self.model.encode(
            texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True
        )
        return embeddings.astype("float32")

    def build_faiss_index(self, embeddings: np.ndarray, docs: List[Document]):
        """Создание FAISS индекса"""
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)  # type: ignore

        # Сохранение документов
        self.doc_store = {doc.metadata["chunk_id"]: doc for doc in docs}

    def save_index(self, index_path: str, docs_path: str):
        """Сохранение индекса и документов"""
        faiss.write_index(self.index, index_path)
        with open(docs_path, "wb") as f:
            pickle.dump(self.doc_store, f)

    def semantic_search(self, query: str, top_k: int = 5):
        """Семантический поиск по индексу"""
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_embedding, top_k)  # type: ignore

        results = []
        for idx, distance in zip(indices[0], distances[0]):
            doc_id = list(self.doc_store.keys())[idx]
            doc = self.doc_store[doc_id]
            results.append({
                "document": doc.page_content,
                "metadata": doc.metadata,
                "score": float(distance)
            })
        return results


if __name__ == "__main__":
    MD_FOLDER = "markdown_docs"
    INDEX_PATH = "C:\\Users\\Evgenii\\Desktop\\OCR_bot\\sent_transformers\\zakaryan_susanna_faiss_index.index"
    DOCS_PATH = "C:\\Users\\Evgenii\\Desktop\\OCR_bot\\sent_transformers\\zakaryan_susanna_documents.pkl"

    processor = MarkdownProcessor()
    indexer = VectorIndexer()

    print("🔄 Обработка Markdown-документов...")
    all_documents = processor.process_markdown_files(MD_FOLDER)
    print(f"✅ Создано {len(all_documents)} чанков")

    print("🔮 Генерация эмбеддингов с SentenceTransformers...")
    embeddings = indexer.create_embeddings(all_documents, batch_size=16)

    print("📊 Создание FAISS индекса...")
    indexer.build_faiss_index(embeddings, all_documents)
    indexer.save_index(INDEX_PATH, DOCS_PATH)
    print(f"🎉 Индекс успешно сохранен: {INDEX_PATH}")

    query = "Какое заключение по делу кредиторов?"
    results = indexer.semantic_search(query)

    print("\n🔍 Результаты поиска:")
    for i, res in enumerate(results):
        print(f"\nРезультат #{i + 1} (Сходство: {res['score']:.3f})")
        print(f"Ключевые слова: {res['metadata']['keywords'][:5]}")
        print(f"Текст: {res['document'][:2000]}...")
