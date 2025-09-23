import os
from dotenv import load_dotenv
from langchain_cerebras import ChatCerebras

def run_simple_cerebras_test():
    """
    Initializes the ChatCerebras client and calls it with a simple prompt
    to capture the model's raw output.
    """
    # Load environment variables from the .env file in the current directory
    load_dotenv()

    # Check if the API key is available
    api_key = os.getenv("CEREBRAS_API_KEY")
    if not api_key:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!! ERROR: Could not find CEREBRAS_API_KEY in your .env file. !!!")
        print("!!! Please ensure the .env file is in the same directory as this script. !!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        return

    # The model name that is causing issues
    model_to_test = "qwen-3-coder-480b"
    simple_prompt = "What is the capital of Nepal?"

    print(f"--- Initializing ChatCerebras with model: {model_to_test} ---")
    
    try:
        # Initialize the ChatCerebras client, just like in your graph.py
        llm = ChatCerebras(model=model_to_test)

        print(f"--- Sending prompt: '{simple_prompt}' ---\n")

        # Invoke the model
        response = llm.invoke(simple_prompt)

        # --- This is the critical part ---
        # We need to see the complete, unmodified string in the .content attribute
        raw_content = response.content

        print("--- RAW RESPONSE RECEIVED ---")
        print("vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv")
        print(raw_content)
        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        print("--- END OF RAW RESPONSE ---")

    except Exception as e:
        print(f"\n--- An Error Occurred During the API Call ---")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Details: {e}")

if __name__ == "__main__":
    run_simple_cerebras_test()