from retrieve_cat.llm import llm_factory
from retrieve_cat.vector_store.wrappers import ChromaWrapper, FaissWrapper

from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from retrieve_cat.vector_store.collection import Collection
import yaml


class RagConfig:
    @classmethod
    def from_file(cls, path):
        with open(path, "r") as f:
            data = yaml.safe_load(f)     
        return cls(data) 

    def __init__(self, data: dict):
        self.data = data 

    @property
    def llm(self):
        return llm_factory.create(self.data["llm"]["type"], **self.data["llm"].get("args", {}))

    @property
    def embedding_model(self):
        return self.data["embedding_model"] 

    @property
    def engine(self):
        rag_engine = self.data["index"]["engine"]
        if rag_engine == "chromadb":
            wrapper = ChromaWrapper
        elif rag_engine == "faiss":
            wrapper = FaissWrapper
        else:
            raise Exception(f"Unknown engine {rag_engine}")

        return wrapper
    
    @property
    def chunk_size(self):
        return self.data["index"]["chunk_size"]

    @property
    def chunk_overlap(self):
        return self.data["index"]["chunk_overlap"]

    @property
    def collection(self):
        embedding_function = SentenceTransformerEmbeddings(model_name=self.embedding_model)
        return Collection.create(self.persistent_path, embedding_function, self.engine)

    @property
    def persistent_path(self):
        return self.data["persistent_path"]