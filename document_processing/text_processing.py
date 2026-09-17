from langchain_community.document_loaders import TextLoader

def text_processor(file):

    loader = TextLoader(file, encoding='utf-8')
    text_doc = loader.load()

    return text_doc