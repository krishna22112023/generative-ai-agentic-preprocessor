from agents import Agent, Runner
import asyncio
from pathlib import Path
import sys

base_path = str(Path(__file__).resolve().parent.parent.parent)
sys.path.append(base_path)

from src.tools.image_processor import image_processor
from configs import settings


system_prompt = """
You are an image preprocessing and annotation agent that takes user input instructions along with the compulsory input arguments, including:
1. input_directory (str): The directory containing the input images.
2. output_directory (str): The directory where the processed images and annotations will be saved.
Optional arguments include:
1. model_id (str) : Model ID for annotation. Default is 'IDEA-Research/grounding-dino-base'.
2. metric (str) : for assessing image quality. Default is 'brisque'.
You will then proceed to call the tool to execute the user's request.
Stop the execution of the tool when you see 'Image processing and annotation complete!'
"""

agent = Agent(
    name="Image processor agent",
    instructions=system_prompt,
    tools=[image_processor],
)

async def conversation_loop():
    print("Welcome to the Image Preprocessing and Annotation Agent!")
    print("Please provide the complete request including input_directory, output_directory, and optionally the huggingface model_id for auto-annotation as well as image quality metric to optimize.")
    conversation_history = []  # Maintains all previous messages for context

    while True:
        user_input = input("User: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting conversation.")
            break

        # Append user message to conversation history.
        conversation_history.append({"role": "user", "content": user_input})

        # Run the agent with the current conversation history.
        result = await Runner.run(agent, input=user_input)
        agent_response = result.final_output
        print("Agent:", agent_response)

        # Append agent response to conversation history.
        conversation_history.append({"role": "assistant", "content": agent_response})

if __name__ == "__main__":
    asyncio.run(conversation_loop())