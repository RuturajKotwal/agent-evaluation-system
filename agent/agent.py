from models import model
from tools import calculate_right_refund, issue_refund_ticket
from tools import get_user_purchase_history, check_current_date
from dotenv import load_dotenv
from langchain.agents import create_agent
from prompts import agent_system_prompt
from objects import Answer
from langgraph.checkpoint.memory import InMemorySaver
from langchain.messages import HumanMessage
from uuid import uuid4

load_dotenv()

agent = create_agent(
    model=model,
    system_prompt=agent_system_prompt,
    tools=[get_user_purchase_history,calculate_right_refund,
           issue_refund_ticket,check_current_date],
    response_format=Answer,
    checkpointer=InMemorySaver()
)

def run_conversation():
    config = {"configurable": {"thread_id": str(uuid4())}}

    text = "I would like to request a refund"
    print(text)
    question = HumanMessage(content=text)
    conversation = True

    while conversation:

        # invoking our agent
        response = agent.invoke({"messages": question},config=config)
        # collecting the structured response
        structured_response = response["structured_response"]

        print(structured_response.answer)

        conversation = structured_response.conversation

        if conversation:
            user_input = input("reply: \n")
            question = HumanMessage(content=user_input)

    print("conversation ended")

run_conversation()