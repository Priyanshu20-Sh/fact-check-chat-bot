
from langgraph.graph import StateGraph, add_messages, START, END
from typing_extensions import Annotated
from typing import List, TypedDict
from langchain_core.messages import AnyMessage
from langgraph.checkpoint.memory import InMemorySaver

from Self_agent_rag.query_re_write import re_write_query
from Self_agent_rag.verification_agent import verify_claims, summarize_answer
from langchain_tavily import TavilySearch
from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from Self_agent_rag.router_agent import router_agent

from dotenv import load_dotenv
load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.5)
hybrid_retriever = None

@tool
def get_rewrite_query(query: str) -> str:
    """Use this tool if the user's query is not clear or lacks detail. It rewrites the query into a highly searchable format."""
    result = re_write_query(query)
    return result

@tool
def get_hybrid_documents(queries: str) -> str:
    """Use this tool to search your local Vector Database for documents. Pass the user's query to get facts."""
    global hybrid_retriever
    if hybrid_retriever is None:
        return "No document is currently loaded. Please ask the user to upload a document."
    docs = hybrid_retriever.invoke(queries)
    return "\n\n".join([d.page_content for d in docs])

@tool
def get_summarize_answer(question: str, context: str) -> str:
    """Use this tool to generate a final answer using a question and the retrieved context."""
    result = summarize_answer(question, context)
    return result

tavily_tool_online_search = TavilySearch(max_results=5, search_depth="advanced")

tools = [
    tavily_tool_online_search, 
    get_rewrite_query, 
    get_hybrid_documents, 
    get_summarize_answer
] 

llm_with_tool = create_agent(
    model=llm,
    tools=tools,
    system_prompt="you are helpful assistant that can answer user properly according to query with precise and accurate and use various tools if you need any tools combination just use",
    checkpointer=InMemorySaver()
)

class state(TypedDict):
    question: str
    messages: Annotated[List[AnyMessage], add_messages]
    route: str                  # router decision: document, web, general
    query_rewritten: str        # rewritten query
    generate_answer: str        # generated answer text
    fact_check_report: str      # output from verification agent
    final_answer: str           # answer shown to user
    retry_count: int            # how many time tried

def node_llm_tool(state):
    msgs = state.get("messages") or [("user", state["question"])]
    res = llm_with_tool.invoke({"messages": msgs})
    final_ans = res["messages"][-1].content if isinstance(res, dict) else res[-1].content
    return {
        "messages": res["messages"] if isinstance(res, dict) else res,
        "final_answer": final_ans
    }

def node_router_result(state):
    route = router_agent(state["question"])
    print(f"\n[router decision ] Selected route {route}")
    return {"route": route}

def node_re_write_query(state):
    rewritten_query = re_write_query(state["question"])
    print("REWRITTEN QUERY", rewritten_query)
    current_retry = state.get("retry_count", 0)
    return {
        "query_rewritten": rewritten_query,
        "retry_count": current_retry + 1
    }

def tavily_node(state:state):
    print("\n[TAVILY] Searching Internet")
    result = tavily_tool_online_search.invoke(state["question"])
    context = str(result)
    return {"generate_answer": context}

def node_general(state: state) -> state:
    print("\n[general] Using only llm ")
    answer = llm.invoke(state["question"]).content
    return {"generate_answer": answer}

def node_hybrid_retrieval(state: state) -> state:
    global hybrid_retriever
    if hybrid_retriever is None:
        print("[Reteriver_docs] 0 (No document loaded)")
        return {"generate_answer": "No document loaded. Please upload a document."}
    
    docs = hybrid_retriever.invoke(state["query_rewritten"])
    print("[RETRIEVED_docs]", len(docs))
    context_str = "\n\n".join([doc.page_content for doc in docs])
    return {"generate_answer": context_str}

def node_verification(state: state) -> state:
    print("\n[VERIFICATION]")
    report = verify_claims(state["question"], state["generate_answer"])
    return {"fact_check_report": report}

def node_summarize(state:state):
    print("\n[SUMMARIZE]")
    # If there is a fact_check_report (document route), summarize it.
    # Otherwise, summarize the generator answer (web/general routes).
    context = state.get("fact_check_report") or state.get("generate_answer")
    summary = summarize_answer(state["question"], context)
    return {"final_answer": summary}

def route_decision(state):
    route = state["route"]
    if route == "document":
        return "query_rewrite"    
    elif route == "web":
        return "tavily_search"    
    elif route == "general":
        return "generator"

graph = StateGraph(state)

graph.add_node("router", node_router_result)
graph.add_node("query_re", node_re_write_query)
graph.add_node("hybrid_retrieval", node_hybrid_retrieval)
graph.add_node("tavily_search", tavily_node)
graph.add_node("general_answer", node_general)
graph.add_node("verification", node_verification)
graph.add_node("agent", node_llm_tool)
graph.add_node("summarize", node_summarize)

graph.add_edge(START, "router")
graph.add_conditional_edges("router", route_decision, {
    "query_rewrite": "query_re",
    "tavily_search": "tavily_search",
    "generator": "general_answer"
})

graph.add_edge("query_re", "hybrid_retrieval")
graph.add_edge("hybrid_retrieval", "verification")
graph.add_edge("verification", "summarize")

graph.add_edge("tavily_search", "summarize")
graph.add_edge("general_answer", "summarize")

graph.add_edge("agent", END)
graph.add_edge("summarize", END)

# Compile
app = graph.compile(checkpointer=InMemorySaver())

if __name__ == "__main__":
    # Test 
    config = {"configurable": {"thread_id": "1"}}
    result = app.invoke({"messages": "Introduction embeddig model from uploaded pdf", "retry_count": 0}, config=config)

