import json
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ruamel.yaml import YAML


def extract_pages_from_pdf(pdf_path):
    docs = []
    loader = PyPDFLoader(pdf_path)
    pages = loader.load_and_split()
    # Skip intro and appendix
    for page in range(17, 460):
        text = pages[page].page_content
        docs.append(text)
    return docs


def split_docs(docs):
    with open("params.yaml") as f:
        params = YAML().load(f)
    split_params = params['TextSplitter']

    text_splitter = RecursiveCharacterTextSplitter(separators=[".\n", ".", "\n"], **split_params)
    return text_splitter.split_text("\n".join(docs))


if __name__ == '__main__':
    pdf_path = 'https://github.com/progit/progit2/releases/download/2.1.426/progit.pdf'
    docs = extract_pages_from_pdf(pdf_path)
    docs = split_docs(docs)

    with open("data/docs.json", "w") as f:
        json.dump(docs, f)
