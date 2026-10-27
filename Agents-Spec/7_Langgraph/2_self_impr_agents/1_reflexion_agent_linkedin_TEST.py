


# model requires more memory than availabel; TEST LATER



from langchain_core.messages import BaseMessage, HumanMessage





from langgraph.graph import END, MessageGraph, StateGraph

from typing import List, Sequence
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


from langchain_ollama import ChatOllama


#-------instanciate language model---------#
llm = ChatOllama(
    model="qwen2.5:3b-instruct",      #"qwen:latest",
    base_url="http://localhost:11434",
)




# Generation prompt
generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a professional LinkedIn content assistant tasked with crafting engaging, insightful, and well-structured LinkedIn posts."
            " Generate the best LinkedIn post possible for the user's request."
            " If the user provides feedback or critique, respond with a refined version of your previous attempts, improving clarity, tone, or engagement as needed.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)


# combining the generation_prompt with a language model (LLM) t
generate_chain = generation_prompt | llm




# Reflection prompt
reflection_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a professional LinkedIn content strategist and thought leadership expert. Your task is to critically evaluate the given LinkedIn post and provide a comprehensive critique. Follow these guidelines:

        1. Assess the post’s overall quality, professionalism, and alignment with LinkedIn best practices.
        2. Evaluate the structure, tone, clarity, and readability of the post.
        3. Analyze the post’s potential for engagement (likes, comments, shares) and its effectiveness in building professional credibility.
        4. Consider the post’s relevance to the author’s industry, audience, or current trends.
        5. Examine the use of formatting (e.g., line breaks, bullet points), hashtags, mentions, and media (if any).
        6. Evaluate the effectiveness of any call-to-action or takeaway.

        Provide a detailed critique that includes:
        - A brief explanation of the post’s strengths and weaknesses.
        - Specific areas that could be improved.
        - Actionable suggestions for enhancing clarity, engagement, and professionalism.

        Your critique will be used to improve the post in the next revision step, so ensure your feedback is thoughtful, constructive, and practical.
        """
    ),
    MessagesPlaceholder(variable_name="messages")
])

# reflect chain
reflect_chain = reflection_prompt | llm




#----------state represents the evolving context of the conversation or task. --------#


"""
State manually
from typing import List, Annotated, TypedDict
from langchain.schema import HumanMessage, AIMessage, SystemMessage

# Define State with TypedDict
class AgentState(TypedDict):
    messages: Annotated[List[HumanMessage | AIMessage | SystemMessage], "add_messages"]
"""
from langgraph.graph import MessageGraph
from typing import List, Annotated, TypedDict


#from langchain.schema import HumanMessage, AIMessage, SystemMessage

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage



# Initialize a predefined MessageGraph
graph = MessageGraph()



#-------Generation and reaction nodes----------------#
def generation_node(state: Sequence[BaseMessage]) -> List[BaseMessage]:
    generated_post = generate_chain.invoke({"messages": state})
    return [AIMessage(content=generated_post.content)]


def reflection_node(messages: Sequence[BaseMessage]) -> List[BaseMessage]:
    res = reflect_chain.invoke({"messages": messages})  # Passes messages as input to reflect_chain
    return [HumanMessage(content=res.content)]  # Returns the refined message as HumanMessage for feedback


# add nodes to the graph
graph.add_node("generate", generation_node)
graph.add_node("reflect", reflection_node)


# edges
graph.add_edge("reflect", "generate") # after refining is done, go back to generation


# ------------ entry-point -------------#
graph.set_entry_point("generate")




#------ router for decision making -----------#


# pre-defiend logic for now
def should_continue(state: List[BaseMessage]):
    print(state)
    print(len(state))
    print("----------------------------------------------------------------------")
    if len(state) > 6:
        return END
    return "reflect"




#-------conditional edge------------#
graph.add_conditional_edges("generate", should_continue)


#------compile workflow----------#
workflow = graph.compile()



#----------defining inputs---------#
inputs = HumanMessage(content="""Write a linkedin post on getting a software developer job at IBM under 160 characters""")



# executing the workflow
response = workflow.invoke(inputs)

print(response[1].content)
print(response[2].content)
print(response[-1].content)



"""
"🌟Ready to code for @IBM? Here’s how: focus on key skills, build projects, and network with professionals in tech communities. #IBMSkills #TechCareer"

Let me know if you'd like any adjustments or additional information!

Absolutely, here’s an even more enhanced version of the LinkedIn post:

"🌟Looking for a software developer role at @IBM? Focus on key skills like Python, Java, and cloud technologies. Build your portfolio with real projects and join tech communities to network with professionals. What are you doing now to land that job? #IBMSkills #TechCareer

💡 What specific steps are you taking in building your technical skills and networking? Let’s discuss how you can take it to the next level! #SoftwareDevelopment"

Feel free to let me know if this version meets your requirements or if you need any further tweaks!
"""


