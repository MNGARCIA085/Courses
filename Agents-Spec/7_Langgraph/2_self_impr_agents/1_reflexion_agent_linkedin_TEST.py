


# model requires more memory than availabel; TEST LATER



from langchain_core.messages import BaseMessage, HumanMessage





from langgraph.graph import END, MessageGraph, StateGraph

from typing import List, Sequence
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


from langchain_ollama import ChatOllama


#-------instanciate language model---------#
llm = ChatOllama(
    model="llama3.1:8b",      #"qwen2.5:3b-instruct",      #"qwen:latest",
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

"""
pr_agents$ 
(env) marcos@marcos-desa:~/Desktop/Courses/Agents-Spec/7_Langgraph/2_self_impr_agents$ 
(env) marcos@marcos-desa:~/Desktop/Courses/Agents-Spec/7_Langgraph/2_self_impr_agents$ 
(env) marcos@marcos-desa:~/Desktop/Courses/Agents-Spec/7_Langgraph/2_self_impr_agents$ ls
'1. lab - Building a Reflection Agent with LangGraph.ipynb'
 1_reflexion_agent_linkedin_TEST.py
'2. lab - Building a Reflexion Agent with External Knowledge In.ipynb'
 2_ref_ext_know_TEST.py
'3. Build Reasoning and Acting AI Agents with LangGraph.ipynb'
 301.py
(env) marcos@marcos-desa:~/Desktop/Courses/Agents-Spec/7_Langgraph/2_self_impr_agents$ python 1_reflexion_agent_linkedin_TEST.py 
/home/marcos/Desktop/Courses/Agents-Spec/7_Langgraph/2_self_impr_agents/1_reflexion_agent_linkedin_TEST.py:105: LangGraphDeprecatedSinceV10: MessageGraph is deprecated in LangGraph v1.0.0, to be removed in v2.0.0. Please use StateGraph with a `messages` key instead. Deprecated in LangGraph V1.0 to be removed in V2.0.
  graph = MessageGraph()
[HumanMessage(content='Write a linkedin post on getting a software developer job at IBM under 160 characters', additional_kwargs={}, response_metadata={}, id='05376afe-4271-46f3-9bf6-7bc80fc8f247'), AIMessage(content='"Unlock your tech potential! Learn how to land a software dev role at IBM: [link to resource] From coding skills to company culture, get insider tips and tricks to boost your chances. #IBM #SoftwareDeveloper #CareerAdvice"', additional_kwargs={}, response_metadata={}, id='cf47852f-3b67-4093-964f-139af40ae22c', tool_calls=[], invalid_tool_calls=[])]
2
----------------------------------------------------------------------
[HumanMessage(content='Write a linkedin post on getting a software developer job at IBM under 160 characters', additional_kwargs={}, response_metadata={}, id='05376afe-4271-46f3-9bf6-7bc80fc8f247'), AIMessage(content='"Unlock your tech potential! Learn how to land a software dev role at IBM: [link to resource] From coding skills to company culture, get insider tips and tricks to boost your chances. #IBM #SoftwareDeveloper #CareerAdvice"', additional_kwargs={}, response_metadata={}, id='cf47852f-3b67-4093-964f-139af40ae22c', tool_calls=[], invalid_tool_calls=[]), HumanMessage(content='', additional_kwargs={}, response_metadata={}, id='2b307c64-b4b0-4de0-bab6-ad7d5fa7cebf'), AIMessage(content='Here\'s another attempt at crafting a LinkedIn post on getting a software developer job at IBM under 160 characters:\n\n"Dreaming of an IBM role? Boost your chances with our top 5 tips: [link to resource] From coding skills to IBM culture, get hired by the best! #IBM #SoftwareDeveloper #CareerTips"', additional_kwargs={}, response_metadata={}, id='dedd39e6-7637-4a77-9924-66fd2519064d', tool_calls=[], invalid_tool_calls=[])]
4
----------------------------------------------------------------------
[HumanMessage(content='Write a linkedin post on getting a software developer job at IBM under 160 characters', additional_kwargs={}, response_metadata={}, id='05376afe-4271-46f3-9bf6-7bc80fc8f247'), AIMessage(content='"Unlock your tech potential! Learn how to land a software dev role at IBM: [link to resource] From coding skills to company culture, get insider tips and tricks to boost your chances. #IBM #SoftwareDeveloper #CareerAdvice"', additional_kwargs={}, response_metadata={}, id='cf47852f-3b67-4093-964f-139af40ae22c', tool_calls=[], invalid_tool_calls=[]), HumanMessage(content='', additional_kwargs={}, response_metadata={}, id='2b307c64-b4b0-4de0-bab6-ad7d5fa7cebf'), AIMessage(content='Here\'s another attempt at crafting a LinkedIn post on getting a software developer job at IBM under 160 characters:\n\n"Dreaming of an IBM role? Boost your chances with our top 5 tips: [link to resource] From coding skills to IBM culture, get hired by the best! #IBM #SoftwareDeveloper #CareerTips"', additional_kwargs={}, response_metadata={}, id='dedd39e6-7637-4a77-9924-66fd2519064d', tool_calls=[], invalid_tool_calls=[]), HumanMessage(content='', additional_kwargs={}, response_metadata={}, id='0a3ec644-8194-41d0-a14b-c78f5dbf6b1e'), AIMessage(content='Here\'s another attempt at crafting a LinkedIn post on getting a software developer job at IBM under 160 characters:\n\n"Want to join @IBMinnovation? Learn our top requirements for software devs: [link to resource] Get the inside scoop and land your dream role! #IBM #SoftwareDeveloper #CareerAdvice"', additional_kwargs={}, response_metadata={}, id='38161f4a-7f1c-4682-8efd-dd0dd63c35e9', tool_calls=[], invalid_tool_calls=[])]
6
----------------------------------------------------------------------
[HumanMessage(content='Write a linkedin post on getting a software developer job at IBM under 160 characters', additional_kwargs={}, response_metadata={}, id='05376afe-4271-46f3-9bf6-7bc80fc8f247'), AIMessage(content='"Unlock your tech potential! Learn how to land a software dev role at IBM: [link to resource] From coding skills to company culture, get insider tips and tricks to boost your chances. #IBM #SoftwareDeveloper #CareerAdvice"', additional_kwargs={}, response_metadata={}, id='cf47852f-3b67-4093-964f-139af40ae22c', tool_calls=[], invalid_tool_calls=[]), HumanMessage(content='', additional_kwargs={}, response_metadata={}, id='2b307c64-b4b0-4de0-bab6-ad7d5fa7cebf'), AIMessage(content='Here\'s another attempt at crafting a LinkedIn post on getting a software developer job at IBM under 160 characters:\n\n"Dreaming of an IBM role? Boost your chances with our top 5 tips: [link to resource] From coding skills to IBM culture, get hired by the best! #IBM #SoftwareDeveloper #CareerTips"', additional_kwargs={}, response_metadata={}, id='dedd39e6-7637-4a77-9924-66fd2519064d', tool_calls=[], invalid_tool_calls=[]), HumanMessage(content='', additional_kwargs={}, response_metadata={}, id='0a3ec644-8194-41d0-a14b-c78f5dbf6b1e'), AIMessage(content='Here\'s another attempt at crafting a LinkedIn post on getting a software developer job at IBM under 160 characters:\n\n"Want to join @IBMinnovation? Learn our top requirements for software devs: [link to resource] Get the inside scoop and land your dream role! #IBM #SoftwareDeveloper #CareerAdvice"', additional_kwargs={}, response_metadata={}, id='38161f4a-7f1c-4682-8efd-dd0dd63c35e9', tool_calls=[], invalid_tool_calls=[]), HumanMessage(content='', additional_kwargs={}, response_metadata={}, id='171ab74c-d7f6-4492-b267-88758d30697d'), AIMessage(content='Here\'s another attempt at crafting a LinkedIn post on getting a software developer job at IBM under 160 characters:\n\n"Dreaming of an @IBMCareers role? Check our latest blog for top tips to become a IBM software dev: [link to resource] #IBM #SoftwareDeveloper #CareerAdvice"', additional_kwargs={}, response_metadata={}, id='3bab940a-eba3-4dbf-bdab-915821a341b0', tool_calls=[], invalid_tool_calls=[])]
8
----------------------------------------------------------------------
"Unlock your tech potential! Learn how to land a software dev role at IBM: [link to resource] From coding skills to company culture, get insider tips and tricks to boost your chances. #IBM #SoftwareDeveloper #CareerAdvice"

Here's another attempt at crafting a LinkedIn post on getting a software developer job at IBM under 160 characters:

"Dreaming of an @IBMCareers role? Check our latest blog for top tips to become a IBM software dev: [link to resource] #IBM #SoftwareDeveloper #CareerAdvice"
"""
