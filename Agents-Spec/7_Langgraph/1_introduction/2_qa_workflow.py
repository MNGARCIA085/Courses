from typing import Optional, TypedDict

from langgraph.graph import END, StateGraph
from langchain_ollama import ChatOllama


# ---------- STATE ----------

class QAState(TypedDict):
    # Stores the user's input question.
    question: Optional[str]

    # Stores relevant context about the guided project.
    context: Optional[str]

    # Stores the generated response.
    answer: Optional[str]

    # Stores whether the input is valid.
    valid: Optional[bool]

    # Stores an error message if validation fails.
    error: Optional[str]


# ---------- NODES ----------

def input_validation_node(state):
    # Extract the question and strip leading/trailing spaces.
    question = state.get("question", "").strip()

    # If the question is empty, return an error.
    if not question:
        return {
            "valid": False,
            "error": "Question cannot be empty.",
        }

    # If the question is valid, return valid status.
    return {
        "valid": True,
    }


def context_provider_node(state):
    question = state.get("question", "").lower()

    # Check if the question is related to the guided project.
    if "langgraph" in question or "guided project" in question:
        context = (
            "This guided project is about using LangGraph, a Python library "
            "to design state-based workflows. LangGraph simplifies building "
            "complex applications by connecting modular nodes with "
            "conditional edges."
        )

        return {
            "context": context,
        }

    # If unrelated, set context to None.
    return {
        "context": None,
    }


# ---------- LLM ----------

llm = ChatOllama(
    model="qwen2.5:3b-instruct",      #"qwen:latest",
    base_url="http://localhost:11434",
)


def llm_qa_node(state):
    # Extract the question and context from the state.
    question = state.get("question", "")
    context = state.get("context", None)

    # Check for missing context and return a fallback response.
    if not context:
        return {
            "answer": "I don't have enough context to answer your question."
        }

    # Construct the prompt dynamically.
    prompt = (
        f"Context: {context}\n"
        f"Question: {question}\n"
        "Answer the question based on the provided context."
    )

    # Generate the response.
    try:
        response = llm.invoke(prompt)

        return {
            "answer": response.content.strip(),
        }

    except Exception as e:
        return {
            "answer": f"An error occurred: {str(e)}",
        }


# ---------- GRAPH ----------

def main():
    # Create the graph.
    qa_workflow = StateGraph(QAState)

    # Add nodes.
    qa_workflow.add_node("InputNode", input_validation_node)
    qa_workflow.add_node("ContextNode", context_provider_node)
    qa_workflow.add_node("QANode", llm_qa_node)

    # Add edges.
    qa_workflow.add_edge("InputNode", "ContextNode")
    qa_workflow.add_edge("ContextNode", "QANode")
    qa_workflow.add_edge("QANode", END)

    # Set entry point.
    qa_workflow.set_entry_point("InputNode")

    # Compile the graph.
    qa_app = qa_workflow.compile()

    # Invoke the application.
    result = qa_app.invoke(
        {
            "question": "What is Probabilist DL?"
        }
    )

    print(result)


    # other
    #r2 = qa_app.invoke({"question": "What is LangGraph?"})
    #print(r2)


if __name__ == "__main__":
    main()



"""
(env) marcos@marcos-HP-ProDesk-600-G1-SFF:~/Escritorio/AI-prod/new_agents$ python 101_21.py
/home/marcos/Escritorio/AI-prod/new_agents/env/lib/python3.10/site-packages/langgraph/checkpoint/base/__init__.py:18: LangChainPendingDeprecationWarning: The default value of `allowed_objects` will change in a future version. Pass an explicit value (e.g., allowed_objects='messages' or allowed_objects='core') to suppress this warning.
  from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
{'question': 'What is the weather today?', 'context': None, 'answer': "I don't have enough context to answer your question.", 'valid': True}
{'question': 'What is LangGraph?', 'context': 'This guided project is about using LangGraph, a Python library to design state-based workflows. LangGraph simplifies building complex applications by connecting modular nodes with conditional edges.', 'answer': 'LangGraph is a Python library designed for designing state-based workflows. It allows users to create complex applications by connecting modular nodes using conditional edges, making it easier and more efficient to build such systems. By simplifying the process of linking different components with decision-making paths, LangGraph 
helps developers manage and scale their workflow applications effectively.', 'valid': True}
"""