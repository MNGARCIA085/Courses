


# langgraph 0.2.57

#--------------STATE--------------#
from typing import TypedDict, Optional

class AuthState(TypedDict):
    username: Optional[str] 
    password: Optional[str]
    is_authenticated: Optional[bool]
    output: Optional[str]





#-----------NODES----------------#



def input_node(state):
    print(state)
    if state.get('username', "") =="":
        username = input("What is your username?")

    password = input("Enter your password: ")
    
    if state.get('username', "") =="":
        return {"username":username, "password":password}
    else:
        return {"password": password}




def validate_credentials_node(state):
    # Extract username and password from the state
    username = state.get("username", "")
    password = state.get("password", "")

    print("Username :", username, "Password :", password)
    # Simulated credential validation
    if username == "test_user" and password == "secure_password":
        is_authenticated = True
    else:
        is_authenticated = False

    # Return the updated state with authentication result
    return {"is_authenticated": is_authenticated}



# Define the success node
def success_node(state):
    return {"output": "Authentication successful! Welcome."}


# Define the failure node
def failure_node(state):
    return {"output": "Not Successfull, please try again!"}


# decision making point in the workflow
def router(state):
    if state['is_authenticated']:
        return "success_node"
    else:
        return "failure_node"



#------------CREATING THE GRAPH-------------#
from langgraph.graph import StateGraph
from langgraph.graph import END

# Create an instance of StateGraph with the GraphState structure
workflow = StateGraph(AuthState)
print(workflow)


# add nodes
workflow.add_node("InputNode", input_node)
workflow.add_node("ValidateCredential", validate_credentials_node)
workflow.add_node("Success", success_node)
workflow.add_node("Failure", failure_node)


# edges

#.. fixed edges
workflow.add_edge("InputNode", "ValidateCredential")
workflow.add_edge("Success", END) 
workflow.add_edge("Failure", "InputNode")



# cond. edge
workflow.add_conditional_edges("ValidateCredential", 
								router, 
								{"success_node": "Success", "failure_node": "Failure"})


"""
start: The node where the conditional edges start (in this case, "ValidateCredential").
router: A function that determines the condition. It checks the current state 
(like the is_authenticated status) and returns 
the appropriate node to transition to (either "Success" or "Failure").
conditions: A dictionary that maps conditions (such as "success_node" or "failure_node") 
to target nodes, indicating where to direct the flow based on the condition
"""


#---Entry point---------#
workflow.set_entry_point("InputNode")






def main():
	""" corr passwd: secure_pasword"""


	# Compiling the workflow
	app = workflow.compile()

	inputs = {"username": "test_user"}
	result = app.invoke(inputs)
	print(result)



if __name__=="__main__":
	main()