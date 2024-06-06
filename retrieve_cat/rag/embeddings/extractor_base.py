from langchain_core.documents import Document

from abc import ABC, abstractmethod

class ExtractorBase(ABC):
    @property
    @abstractmethod
    def documents(self) -> list[Document]:
        pass

    def to_vector(self, embedding_function: callable):
        return embedding_function(self.document)

