import os
import json
from typing import List

from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    ToolMessage,
    BaseMessage,
)

from langgraph.graph import END, MessageGraph

from langchain_ollama import ChatOllama
from langchain_community.tools import DuckDuckGoSearchResults


# ============================================================
# LLM
# ============================================================

llm = ChatOllama(
    model="qwen2.5:3b-instruct", #"llama3.1:latest", #"qwen2.5:3b-instruct",
    base_url="http://localhost:11434",
)


question = "Any ideas for a healthy breakfast"

response = llm.invoke(question)
print(response.content)


# ============================================================
# PROMPT
# ============================================================

prompt_template = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are Dr. Paul Saladino, "Carnivore MD", advocating for
animal-based nutrition and challenging plant-centric dietary dogma.

Your response must:

1. Answer the user's question.
2. Present the evolutionary and biochemical rationale for
   animal-based nutrition.
3. Discuss nutrient density, bioavailability, and plant
   defense compounds.
4. Reflect critically on your answer.
5. Provide 1-3 search queries for additional research.

Focus on:
- organ meats
- carnivore elimination protocol
- bioavailability
- nutrient density
- plant antinutrients
- evolutionary nutrition

{first_instruction}
"""
    ),
    MessagesPlaceholder(variable_name="messages"),
])


# ============================================================
# STRUCTURED OUTPUT
# ============================================================

class Reflection(BaseModel):
    missing: str = Field(
        description="What important information is missing."
    )

    superfluous: str = Field(
        description="What information is unnecessary."
    )


class AnswerQuestion(BaseModel):
    answer: str = Field(
        description="The complete answer to the user's question."
    )

    reflection: Reflection = Field(
        description="A critical reflection on the answer."
    )

    search_queries: List[str] = Field(
        description="One to three search queries for additional research."
    )


class ReviseAnswer(AnswerQuestion):
    references: List[str] = Field(
        description="References supporting the revised answer."
    )


# ============================================================
# INITIAL RESPONDER
# ============================================================

first_responder_prompt = prompt_template.partial(
    first_instruction="""
Provide a detailed answer of approximately 50 words.

Return exactly ONE AnswerQuestion tool call.

Do NOT create a field called "question".
Do NOT create multiple AnswerQuestion tool calls.
"""
)

initial_chain = first_responder_prompt | llm.bind_tools(
    tools=[AnswerQuestion]
)


# ============================================================
# TEST INITIAL RESPONDER
# ============================================================

response = initial_chain.invoke({
    "messages": [
        HumanMessage(content=question)
    ]
})

print("\n--- Full Structured Output ---")
print(response.tool_calls)


# ------------------------------------------------------------
# Validate the tool call
# ------------------------------------------------------------

if not response.tool_calls:
    raise ValueError(
        "Qwen did not return an AnswerQuestion tool call."
    )

if len(response.tool_calls) != 1:
    print(
        f"Warning: Qwen returned {len(response.tool_calls)} "
        "tool calls. Using the first one."
    )

tool_call = response.tool_calls[0]

print("\n--- Tool Name ---")
print(tool_call["name"])

print("\n--- Tool Arguments ---")
print(tool_call["args"])


args = tool_call["args"]

required_fields = {
    "answer",
    "reflection",
    "search_queries",
}

missing_fields = required_fields - set(args.keys())

if missing_fields:
    raise ValueError(
        f"Invalid structured output from Qwen.\n"
        f"Missing fields: {missing_fields}\n"
        f"Received: {args}"
    )


print("\n--- Initial Answer ---")
print(args["answer"])

print("\n--- Reflection ---")
print(args["reflection"])

print("\n--- Search Queries ---")
print(args["search_queries"])


# ============================================================
# SEARCH TOOL
# ============================================================

search_tool = DuckDuckGoSearchResults(
    max_results=3
)


# ============================================================
# TOOL EXECUTION
# ============================================================

def execute_tools(
    state: List[BaseMessage],
) -> List[BaseMessage]:

    last_ai_message = state[-1]

    tool_messages = []

    for tool_call in last_ai_message.tool_calls:

        tool_name = tool_call["name"]

        if tool_name not in [
            "AnswerQuestion",
            "ReviseAnswer",
        ]:
            continue

        call_id = tool_call["id"]

        search_queries = tool_call["args"].get(
            "search_queries",
            []
        )

        query_results = {}

        for query in search_queries:

            print(f"\nSearching: {query}")

            result = search_tool.invoke(query)

            query_results[query] = result

        tool_messages.append(
            ToolMessage(
                content=json.dumps(
                    query_results,
                    default=str
                ),
                tool_call_id=call_id,
            )
        )

    return tool_messages


# ============================================================
# REVISOR
# ============================================================

revise_instructions = """
Revise the previous answer using the new search information.

Apply a rigorous evidence-based approach.

You MUST:

- improve factual accuracy
- distinguish correlation from causation
- acknowledge limitations in the evidence
- avoid unsupported speculation
- incorporate useful information from the search results
- consider individual variability
- keep the answer under 250 words
- include references supporting the revised answer

Return exactly ONE ReviseAnswer tool call.

Do NOT create a field called "question".
Do NOT create multiple ReviseAnswer calls.
"""

revisor_prompt = prompt_template.partial(
    first_instruction=revise_instructions
)

revisor_chain = revisor_prompt | llm.bind_tools(
    tools=[ReviseAnswer]
)


# ============================================================
# GRAPH
# ============================================================

MAX_ITERATIONS = 2


def event_loop(state: List[BaseMessage]) -> str:

    # Count completed search executions
    tool_visits = sum(
        isinstance(message, ToolMessage)
        for message in state
    )

    if tool_visits >= MAX_ITERATIONS:
        return END

    # If the latest AI message has search queries,
    # perform another search/revision cycle.
    last_message = state[-1]

    if isinstance(last_message, AIMessage):

        if last_message.tool_calls:

            for tool_call in last_message.tool_calls:

                queries = tool_call["args"].get(
                    "search_queries",
                    []
                )

                if queries:
                    return "execute_tools"

    return END


graph = MessageGraph()

graph.add_node(
    "respond",
    initial_chain
)

graph.add_node(
    "execute_tools",
    execute_tools
)

graph.add_node(
    "revisor",
    revisor_chain
)

graph.add_edge(
    "respond",
    "execute_tools"
)

graph.add_edge(
    "execute_tools",
    "revisor"
)

graph.add_conditional_edges(
    "revisor",
    event_loop
)

graph.set_entry_point(
    "respond"
)


# ============================================================
# COMPILE
# ============================================================

app = graph.compile()


# ============================================================
# RUN AGENT
# ============================================================

question = """
I'm pre-diabetic and need to lower my blood sugar,
and I have heart issues.

What breakfast foods should I eat and avoid?
"""


responses = app.invoke(question)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n\n================================================")
print("INITIAL ANSWER")
print("================================================")

for message in responses:

    if getattr(message, "tool_calls", None):

        for tool_call in message.tool_calls:

            answer = tool_call["args"].get("answer")

            if answer:
                print(answer)
                break


print("\n\n================================================")
print("ALL ANSWERS")
print("================================================")

answers = []

for message in responses:

    if getattr(message, "tool_calls", None):

        for tool_call in message.tool_calls:

            answer = tool_call["args"].get("answer")

            if answer:
                answers.append(answer)


for i, answer in enumerate(answers):

    print(f"\n--- Answer {i + 1} ---")
    print(answer)