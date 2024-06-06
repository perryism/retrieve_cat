from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma

class ChromaWrapper:
    """
    Wrapper around Chroma

    """
    def __init__(self, embedding_function: Embeddings, path: str):
        # path is a directory
        self.path = path
        self.embedding_function = embedding_function

    def persist(self, docs: list[Document]):
        return Chroma.from_documents(docs, self.embedding_function, persist_directory=self.path)

    def similarity_search(self, query, *args, **kwargs):
        client = Chroma(persist_directory=self.path, embedding_function=self.embedding_function)
        return client.similarity_search(query, *args, **kwargs)

from langchain_community.vectorstores import FAISS
import pickle
class FaissWrapper:
    def __init__(self, embedding_function: Embeddings, path: str):
        self.path = path
        self.embedding_function = embedding_function

    def persist(self,  docs: list[Document]):
        db = FAISS.from_documents(docs, self.embedding_function)
        with open(self.path, 'wb') as f:
            pickle.dump(db, f)
        return db

    def similarity_search(self, query, *args, **kwargs):
        with open(args.database, 'rb') as f:
            faiss = pickle.loads(f.read())
        return faiss.similarity_search(query, *args, **kwargs)
