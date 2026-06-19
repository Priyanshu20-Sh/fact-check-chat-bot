
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq


llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.5)

from dotenv import load_dotenv
load_dotenv()

parser = StrOutputParser()

prompt = PromptTemplate(
    template="""You are an expert route selector for an Autonomous AI Agent.

Your task is to determine the best path for answering the user's question.

User Question:
{question}

Routes:
1. document - Use this for static, archived, local organizational knowledge or specific documents in our database.
2. web - Use this for current events, news, real-time facts, or general web searches.
3. general - Use this for greetings, conversational chit-chat, or general knowledge that doesn't need external search or document database.

Instructions:
- Analyze the question's topic, context, and timeframe.
- Output ONLY the chosen route name in lowercase (e.g., document, web, or general).
- Return absolutely NO other text, reasoning, quotes, or punctuation.

Route:""",
    input_variables=["question"]
)

router_chain = prompt | llm | parser

def router_agent(question: str):
    router_result = router_chain.invoke({"question": question})
    return router_result.strip().lower()
