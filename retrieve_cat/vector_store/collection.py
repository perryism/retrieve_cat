import logging
from langchain_core.documents import Document
from retrieve_cat.rag.embeddings import TextExtractor, PdfExtractor

logger = logging.getLogger(__name__)

class Collection:
    @classmethod
    def create(cls, path, embedding_function, engine_wrapper):
        return cls(path, embedding_function, engine_wrapper)

    def __init__(self, path, embedding_function, engine_wrapper):
        self.path = path
        self.embedding_function = embedding_function
        self.engine_wrapper = engine_wrapper

    def to_documents(self, filepath, chunk_size, chunk_overlap) -> list[Document]:
        # if filepath extension is txt
        if filepath.endswith(".txt"):
            extractor = TextExtractor(filepath, chunk_size, chunk_overlap)
        elif filepath.endswith(".pdf"):
            extractor = PdfExtractor(filepath)
        else:
            raise ValueError("Unsupported file extension")
        return extractor.documents

    def ingest(self, filepath, chunk_size, chunk_overlap):
        # https://python.langchain.com/docs/integrations/vectorstores/chroma/
        docs = self.to_documents(filepath, chunk_size, chunk_overlap)
        logger.info(f"ingesting {len(docs)} documents")
        logger.info(f"db path: {self.path}")
        return self.engine_wrapper(self.embedding_function, self.path).persist(docs)

    def query(self, query, n_results=1):
        db = self.engine_wrapper(self.embedding_function, self.path)
        docs = db.similarity_search(query, k=n_results)
        return [ doc.page_content for doc in docs]
