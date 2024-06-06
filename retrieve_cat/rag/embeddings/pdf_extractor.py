from langchain_community.document_loaders import PyPDFLoader
from .extractor_base import ExtractorBase
from langchain_core.documents import Document

class PdfExtractor(ExtractorBase):
    def __init__(self, filepath):
        self.filepath = filepath

    @property
    def documents(self) -> list[Document]:
        loader = PyPDFLoader(self.filepath)
        return loader.load_and_split()