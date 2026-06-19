from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()
from langchain_core.output_parsers import StrOutputParser
import os

chat_model = ChatGroq(model="llama-3.1-8b-instant", temperature=0.5)

prompt = PromptTemplate(
    template="""
You are an expert query rewriting system for Retrieval-Augmented Generation (RAG).

Your job is to transform  user questions into highly searchable retrieval queries.

Instructions:
1. Preserve the original intent exactly.
2. Expand abbreviations when relevant.
3. Add important technical keywords likely to appear in documents.
4. Replace ambiguous words with precise terminology.
5. Include related concepts if they improve retrieval.
6. Do NOT answer the question.
7. Do NOT explain your reasoning.
8. Return ONLY the rewritten query.
9. Keep the rewritten query under 30 words.

User Query:
{query}

Re_Query:
""",
    input_variables=["query"]
)

parser = StrOutputParser()

chain = prompt | chat_model | parser
def re_write_query(user_query:str):
    re_write_query_result = chain.invoke({"query":user_query})
    return re_write_query_result
