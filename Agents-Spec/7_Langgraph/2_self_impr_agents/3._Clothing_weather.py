
from langchain_community.tools import DuckDuckGoSearchResults



# ============================================================
# SEARCH TOOL
# ============================================================

search_tool = DuckDuckGoSearchResults(
    max_results=3
)


# search_tool.invoke("What's the weather like in Tokyo today?")



# clot. rec. tool

from langchain_core.tools import tool

@tool
def recommend_clothing(weather: str) -> str:
    """
    Returns a clothing recommendation based on the provided weather description.

    This function examines the input string for specific keywords or temperature indicators 
    (e.g., "snow", "freezing", "rain", "85°F") to suggest appropriate attire. It handles 
    common weather conditions like snow, rain, heat, and cold by providing simple and practical 
    clothing advice.

    :param weather: A brief description of the weather (e.g., "Overcast, 64.9°F")
    :return: A string with clothing recommendations suitable for the weather
    """
    weather = weather.lower()
    if "snow" in weather or "freezing" in weather:
        return "Wear a heavy coat, gloves, and boots."
    elif "rain" in weather or "wet" in weather:
        return "Bring a raincoat and waterproof shoes."
    elif "hot" in weather or "85" in weather:
        return "T-shirt, shorts, and sunscreen recommended."
    elif "cold" in weather or "50" in weather:
        return "Wear a warm jacket or sweater."
    else:
        return "A light jacket should be fine."





# tool registry
tools=[search_tool,recommend_clothing]

tools_by_name={ tool.name:tool for tool in tools}



#-------MODEL
from langchain_ollama import ChatOllama
from langchain_core.tools import tool

# Make sure you have Ollama running locally and have pulled the model:
model = ChatOllama(
    model="qwen2.5:3b-instruct", #"qwen2.5:3b-instruct", #"llama3.1:latest", #"qwen2.5:3b-instruct",
    base_url="http://localhost:11434",
)


# system prompt
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage,SystemMessage

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are a helpful AI assistant that thinks step-by-step and uses tools when needed.

When responding to queries:
1. First, think about what information you need
2. Use available tools if you need current data or specific capabilities  
3. Provide clear, helpful responses based on your reasoning and any tool results

Always explain your thinking process to help users understand your approach.
"""),
    MessagesPlaceholder(variable_name="scratch_pad")
])




# binding tools to the model
model_react=chat_prompt|model.bind_tools(tools)


# state
from typing import (Annotated,Sequence,TypedDict)
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    """The state of the agent."""

    # add_messages is a reducer
    # See https://langchain-ai.github.io/langgraph/concepts/low_level/#reducers
    messages: Annotated[Sequence[BaseMessage], add_messages]


"""
deostrates state management
# Example conversation flow:
state: AgentState = {"messages": []}

# append a message using the reducer properly
state["messages"] = add_messages(state["messages"], [HumanMessage(content="Hi")])
print("After greeting:", state["messages"])

# add another message (e.g. a question)
state["messages"] = add_messages(state["messages"], [HumanMessage(content="Weather in NYC?")])
print("After question:", state)
"""


# manual reacxt exec (to undertsbad the flow)

"""

# 1. query proc.
dummy_state: AgentState = {
    "messages": [HumanMessage( "What's the weather like in Zurich, and what should I wear based on the temperature?")]}

response = model_react.invoke({"scratch_pad":dummy_state["messages"]})

dummy_state["messages"]=add_messages(dummy_state["messages"],[response])


# 2. tool exec
tool_call = response.tool_calls[-1]
print("Tool call:", tool_call)

tool_result = tools_by_name[tool_call["name"]].invoke(tool_call["args"])
print("Tool result preview:", tool_result[0]['title'])

tool_message = ToolMessage(
    content=json.dumps(tool_result),
    name=tool_call["name"],
    tool_call_id=tool_call["id"]
)
dummy_state["messages"] = add_messages(dummy_state["messages"], [tool_message])


# 3. proc. eults adn next action
response = model_react.invoke({"scratch_pad": dummy_state["messages"]})
dummy_state['messages'] = add_messages(dummy_state['messages'], [response])

# check if the model wants to use another tool
if response.tool_calls:
    tool_call = response.tool_calls[0]
    tool_result = tools_by_name[tool_call["name"]].invoke(tool_call["args"])
    tool_message = ToolMessage(
        content=json.dumps(tool_result),
        name=tool_call["name"],
        tool_call_id=tool_call["id"]
    )
    dummy_state['messages'] = add_messages(dummy_state['messages'], [tool_message])


# 4. final response gen.
response = model_react.invoke({"scratch_pad": dummy_state["messages"]})
print("Final response generated:", response.content is not None)
print("More tools needed:", bool(response.tool_calls))


Tool call: {'name': 'duckduckgo_results_json', 'args': {'query': 'weather in Zurich'}, 'id': 'ebc5b96f-da12-471f-a32f-ffe79307cbf6', 'type': 'tool_call'}
Traceback (most recent call last):
  File "/home/marcos/Escritorio/AI-prod/Courses/Agents-Spec/7_Langgraph/2_self_impr_agents/3._Clothing_weather.py", line 139, in <module>
    print("Tool result preview:", tool_result[0]['title'])
TypeError: string indices must be integers

"""



#--------with GRAPHS


import json

def tool_node(state: AgentState):
    """Execute all tool calls from the last message in the state."""
    outputs = []
    for tool_call in state["messages"][-1].tool_calls:
        tool_result = tools_by_name[tool_call["name"]].invoke(tool_call["args"])
        outputs.append(
            ToolMessage(
                content=json.dumps(tool_result),
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
            )
        )
    return {"messages": outputs}


def call_model(state: AgentState):
    """Invoke the model with the current conversation state."""
    response = model_react.invoke({"scratch_pad": state["messages"]})
    return {"messages": [response]}


def should_continue(state: AgentState):
    """Determine whether to continue with tool use or end the conversation."""
    messages = state["messages"]
    last_message = messages[-1]
    # If there is no function call, then we finish
    if not last_message.tool_calls:
        return "end"
    # Otherwise if there is, we continue
    else:
        return "continue"


from langgraph.graph import StateGraph, END

# Define a new graph
workflow = StateGraph(AgentState)

# Define the two nodes we will cycle between
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

# Add edges between nodes
workflow.add_edge("tools", "agent")  # After tools, always go back to agent

# Add conditional logic
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "tools",  # If tools needed, go to tools node
        "end": END,          # If done, end the conversation
    },
)

# Set entry point
workflow.set_entry_point("agent")

# Compile the graph
graph = workflow.compile()


"""
visualizing
from IPython.display import Image, display

try:
    display(Image(graph.get_graph().draw_mermaid_png()))
except Exception:
    # This requires some extra dependencies and is optional
    pass
"""


def print_stream(stream):
    """Helper function for formatting the stream nicely."""
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

inputs = {"messages": [HumanMessage(content="What's the weather like in Zurich, and what should I wear based on the temperature?")]}

print_stream(graph.stream(inputs, stream_mode="values"))



"""
/home/marcos/Escritorio/AI-prod/Courses/Agents-Spec/7_Langgraph/2_self_impr_agents/3._Clothing_weather.py:2: DeprecationWarning: `langchain-community` is being sunset and is no longer actively maintained. See https://github.com/langchain-ai/langchain-community/issues/674 for details and migration guidance toward standalone integration packages.
  from langchain_community.tools import DuckDuckGoSearchResults
================================ Human Message =================================

What's the weather like in Zurich, and what should I wear based on the temperature?
================================== Ai Message ==================================

To provide you with the current weather in Zurich and clothing recommendations based on that, we first need to search for the current weather conditions there. Let's use a DuckDuckGo search for this information.
Tool Calls:
  duckduckgo_results_json (965b9462-3147-4e2c-bda3-9e85be22c482)
 Call ID: 965b9462-3147-4e2c-bda3-9e85be22c482
  Args:
    query: current weather in Zurich
================================= Tool Message =================================
Name: duckduckgo_results_json

"snippet: Z\u00fcrich weather today, hour by hour: temperature, chance of rain, wind, UV and air quality, plus sunrise and sunset times., title: Z\u00fcrich Weather Today - Hourly Forecast | Ease Weather, link: https://www.easeweather.com/europe/switzerland/zurich/today, snippet: Weather and weather forecast for Zurich, Switzerland. \u2600\ufe0f\ud83c\udf27\ufe0f Worldwide current weather and 14-day weather forecast, title: Weather Zurich, Switzerland - 14 Days - \ufe0f Worldmeteo, link: https://www.worldmeteo.info/en/europe/switzerland/zurich/weather-100514/, snippet: Zurich Weather Forecast. Providing a local hourly Zurich weather forecast of rain, sun, wind, humidity and temperature. The Long-range 16 day forecast also includes detail for Zurich weather today. Live weather reports from Zurich weather stations and weather warnings that include risk of thunder, high UV index and forecast gales., title: Zurich Weather Forecast, link: https://www.weather-forecast.com/locations/Zurich/forecasts/latest, snippet: Latest weather forecast for Zurich for today's, hourly weather forecast, including today's temperatures in Zurich, wind, rain and more., title: Zurich local weather (live): today, hourly weather, link: https://www.weather25.com/europe/switzerland/zurich?page=today"
================================== Ai Message ==================================

The current weather conditions in Zurich are not directly provided by the search results. For a more accurate forecast and to get detailed temperature information for your clothing recommendation, I will conduct another DuckDuckGo search specifically for Zurich's current weather.

"""