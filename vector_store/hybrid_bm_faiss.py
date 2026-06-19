from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from dotenv import load_dotenv
load_dotenv()
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings




model =  HuggingFaceEmbeddings(model_name = "BAAI/bge-base-en-v1.5")


def hybrid_docs(list_chunks_document):
    list_chunks_document = [doc for doc in list_chunks_document if doc.page_content and doc.page_content.strip()]
    if not list_chunks_document:
        raise ValueError("scanned image not allowed ")
    

    vector_store = FAISS.from_documents(documents=list_chunks_document,embedding=model,)
    f_reteriver = vector_store.as_retriever(search_kwargs={"k": 4})

    b_reteriver = BM25Retriever.from_documents(documents=list_chunks_document)
    b_reteriver.k = 4
    

    hybrid_vector_store= EnsembleRetriever(retrievers=[b_reteriver,f_reteriver],weights=[0.6,0.4])


    return hybrid_vector_store

