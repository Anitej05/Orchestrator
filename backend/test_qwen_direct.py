
import os
import base64
import logging
from openai import OpenAI
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load env vars
load_dotenv()

# Configuration
API_KEY = os.getenv("OLLAMA_API_KEY")
BASE_URL = "https://ollama.com/v1"
MODEL = "qwen3-vl:235b-cloud"

# Real screenshot path
SCREENSHOT_PATH = r"D:\Internship\Orbimesh\backend\storage\browser_screenshots\screenshot_1766475390_7487c933.png"

def get_image_b64():
    with open(SCREENSHOT_PATH, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def test_qwen_direct():
    print(f"🚀 Testing {MODEL} at {BASE_URL}")
    print(f"📁 Using image: {SCREENSHOT_PATH}")
    
    if not API_KEY:
        print("❌ Error: OLLAMA_API_KEY not found in .env")
        return

    client = OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL
    )

    
    # Tool Definition
    tools = [
        {
            "type": "function",
            "function": {
                "name": "browser_action",
                "description": "Execute a browser action based on the visual analysis",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reasoning": {
                            "type": "string",
                            "description": "Detailed reasoning for why this action is chosen"
                        },
                        "action_name": {
                            "type": "string",
                            "enum": ["click", "type", "scroll", "search", "navigate"],
                            "description": "The action to perform"
                        },
                        "params": {
                            "type": "object",
                            "description": "Parameters for the action (e.g., mark number, text to type)"
                        },
                        "confidence": {
                            "type": "number",
                            "description": "Confidence score 0-1"
                        }
                    },
                    "required": ["reasoning", "action_name", "confidence"]
                }
            }
        }
    ]

    try:
        print("\n📤 Sending Tool Call request...")
        img_b64 = get_image_b64()
        
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Analyze the screenshot and call the correct browser action."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_b64}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=1000,
            temperature=0.1,
            tools=tools,
            tool_choice="auto"
        )

        print("\n✅ Response Received!")
        with open("qwen_response.log", "a", encoding="utf-8") as f:
            f.write(f"\nResponse Object: {response}\n")

        if response.choices:
            choice = response.choices[0]
            print(f"Finish Reason: {choice.finish_reason}")
            
            if choice.message.tool_calls:
                print("🛠️  Tool Calls Found!")
                with open("qwen_response.log", "a", encoding="utf-8") as f:
                    f.write("\n--- Tool Calls ---\n")
                for tc in choice.message.tool_calls:
                    print(f"   Function: {tc.function.name}")
                    print(f"   Args: {tc.function.arguments}")
                    with open("qwen_response.log", "a", encoding="utf-8") as f:
                        f.write(f"Function: {tc.function.name}\nArgs: {tc.function.arguments}\n")
            else:
                print("❌ No tool calls found.")
                print(f"Content: {choice.message.content}")
                with open("qwen_response.log", "a", encoding="utf-8") as f:
                    f.write(f"\nContent: {choice.message.content}\n")
                
                if hasattr(choice.message, 'reasoning'):
                    print(f"Reasoning: {choice.message.reasoning}")
                    with open("qwen_response.log", "a", encoding="utf-8") as f:
                        f.write(f"\nReasoning: {choice.message.reasoning}\n")

    except Exception as e:
        print(f"❌ Error during request: {e}")
        with open("qwen_response.log", "a", encoding="utf-8") as f:
            f.write(f"\n❌ Error: {e}\n")

    print("🏁 Test Finished")

if __name__ == "__main__":
    # Clear log file
    with open("qwen_response.log", "w", encoding="utf-8") as f:
        f.write("Starting Test\n")
    test_qwen_direct()

if __name__ == "__main__":
    test_qwen_direct()
