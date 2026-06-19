from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langchain_core.output_parsers import StrOutputParser
import os

# Initialize the LLM
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

# Initialize Tavily search tool
tavily_tool = TavilySearch(max_results=3, search_depth="advanced")

# Prompt for Fact-Checking
fact_check_prompt = PromptTemplate(
    template="""You are a strict Fact-Checking Agent. Your job is to act as a "Truth Layer".
    
You are given a user's QUESTION, the local PDF CONTEXT which might contain OUTDATED or FAKE statistics/lies, and LIVE WEB FACTS.
Your task is to:
1. Extract the claims from the PDF CONTEXT that relate to the QUESTION.
2. Cross-reference them against the LIVE WEB FACTS.
3. Report the claims as VERIFIED (matches data), INACCURATE (e.g., outdated stats), or FALSE (no evidence found or directly contradicted).
4. Provide the correct "real" facts based on the LIVE WEB FACTS.

QUESTION: {question}

PDF CONTEXT (may contain lies):
{context}

LIVE WEB FACTS:
{web_facts}

Provide a structured Fact-Check Report:
""",
    input_variables=["question", "context", "web_facts"],
)

def verify_claims(question: str, context: str) -> str:
    """Verifies claims found in the PDF context against the live web."""
    print("\n[VERIFICATION AGENT] Searching live web to verify claims...")
    try:
        # Search the web for real facts based on the user's question
        web_results = tavily_tool.invoke(question)
        web_facts = str(web_results)
    except Exception as e:
        print(f"[VERIFICATION AGENT] Tavily search error: {e}")
        web_facts = "Web search failed. Unable to verify."
    
    # Run the fact-checking LLM
    print("\n[VERIFICATION AGENT] Cross-referencing PDF claims with live web data...")
    chain = fact_check_prompt | llm | StrOutputParser()
    report = chain.invoke({
        "question": question,
        "context": context,
        "web_facts": web_facts
    })
    
    return report

def summarize_answer(question: str, context: str) -> str:
    """Summarizes the final answer or fact-check report for the user."""
    prompt = PromptTemplate(
        template="""You are a helpful assistant. Synthesize a final, concise answer based on the following Fact-Check Report or context. 
Make sure the user clearly knows if the document contained false information, what the real facts are, or summarize the general answer.

User Question: {question}

Context / Fact-Check Report:
{context}

Final Answer:""",
        input_variables=["question", "context"]
    )
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"question": question, "context": context})
