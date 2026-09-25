from datetime import datetime
from typing import Literal

from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool

from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()


@tool
def calculator(
    a: float,
    b: float,
    operation: Literal["add", "subtract", "multiply", "divide"]
) -> float:

    if operation == "add":
        return a + b

    elif operation == "subtract":
        return a - b

    elif operation == "multiply":
        return a * b

    elif operation == "divide":

        if b == 0:
            raise ValueError("Cannot divide by zero.")

        return a / b

    raise ValueError("Invalid operation.")


@tool
def get_current_date() -> str:

    return datetime.now().strftime("%Y-%m-%d")


@tool
def word_counter(text: str) -> int:

    return len(text.split())



tools = [
    calculator,
    get_current_date,
    word_counter
]



model = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)

model_with_tools = model.bind_tools(tools)



SYSTEM_PROMPT = """
You are a helpful multi-tool AI agent.

You have access to three tools:

1. calculator
   - Performs addition, subtraction, multiplication and division.

2. get_current_date
   - Returns today's date.

3. word_counter
   - Counts the number of words in a text.

Rules:

- Use calculator for mathematical calculations.
- Use get_current_date for questions about today's date.
- Use word_counter when the user asks to count words.
- You can use multiple tools for one request.
- Never invent a tool result.
- After getting the tool results, give the user a clear final answer.
"""



def agent_node(state: MessagesState):

    messages = [
        SystemMessage(content=SYSTEM_PROMPT)
    ] + state["messages"]

    response = model_with_tools.invoke(messages)

    return {
        "messages": [response]
    }



tool_node = ToolNode(tools)



def route_after_agent(state: MessagesState):

    last_message = state["messages"][-1]

    if getattr(last_message, "tool_calls", None):

        return "tools"


    return END



builder = StateGraph(MessagesState)


builder.add_node("agent", agent_node)
builder.add_node("tools", tool_node)



builder.add_edge(
    START,
    "agent"
)


builder.add_conditional_edges(
    "agent",
    route_after_agent
)


builder.add_edge(
    "tools",
    "agent"
)



checkpointer = InMemorySaver()




graph = builder.compile(
    checkpointer=checkpointer
)


def run_agent(user_input, thread_id="student-demo"):

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    result = graph.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": user_input
                }
            ]
        },
        config
    )

    return result["messages"][-1].content

 

if __name__ == "__main__":

    print("=" * 60)
    print(" LANGGRAPH MULTI-TOOL AI AGENT")
    print("=" * 60)

    print("\nAvailable tools:")
    print("1. Calculator")
    print("2. Current Date")
    print("3. Word Counter")

    print("\nType 'exit' to stop.")

    while True:

        user_input = input("\nYou: ")

        if user_input.lower() == "exit":

            print("Agent: Goodbye!")
            break

        try:

            answer = run_agent(user_input)

            print("\nAgent:", answer)

        except Exception as e:

            print("\nError:", e)