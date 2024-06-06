from .extractor_base import ExtractorBase
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain.text_splitter import CharacterTextSplitter

class TextExtractor(ExtractorBase):
    def __init__(self, filepath, chunk_size, chunk_overlap):
        self.filepath = filepath
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @property
    def documents(self) -> list[Document]:
        loader = TextLoader(self.filepath)
        documents = loader.load()

        # split it into chunks
        text_splitter = CharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        return text_splitter.split_documents(documents)