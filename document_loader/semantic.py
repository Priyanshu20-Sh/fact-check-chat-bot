from  langchain_experimental.text_splitter  import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings


embedding_model = HuggingFaceEmbeddings(model_name = "BAAI/bge-base-en-v1.5")

def semantic(docs):
    chunk = SemanticChunker(embeddings=embedding_model, breakpoint_threshold_amount=0.8)
    try:
        list_chunks_document = chunk.split_documents(documents=docs)
    except Exception as e:
        print(f" SemanticChunker failed {e}  Using raw documents.")
        list_chunks_document = []
    
    # Fallback to raw documents if no chunks were created
    if not list_chunks_document:
        list_chunks_document = docs
        
    return list_chunks_document

