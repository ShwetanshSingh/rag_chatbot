from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

class LoadDocument:
    def load_document(document_name="the_wonderful_wizard_of_oz.txt"):
        loader = DirectoryLoader("./data", glob=document_name, use_multithreading=True, show_progress=True)
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        return text_splitter.split_documents(docs)  # splits of the document
                                                    # to be added to vector store