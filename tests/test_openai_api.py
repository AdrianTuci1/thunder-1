import requests
import json
import uuid

def test_openai_api():
    url = "http://localhost:8000/v1/chat/completions"
    
    payload = {
        "model": "thunder-v1",
        "messages": [
            {"role": "user", "content": "Explain the concept of entropy in information theory."}
        ],
        "mode": "thinking",
        "stream": False
    }
    
    print("--- Testing OpenAI API (Non-Streaming) ---")
    try:
        response = requests.post(url, json=payload)
        print(f"Status: {response.status_code}")
        print(json.dumps(response.json(), indent=2))
    except Exception as e:
        print(f"Error: {e}")

    print("\n--- Testing OpenAI API (Streaming) ---")
    payload["stream"] = True
    try:
        response = requests.post(url, json=payload, stream=True)
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                if decoded_line.startswith("data: "):
                    data_str = decoded_line[6:]
                    if data_str == "[DONE]":
                        print("\n[Stream Complete]")
                        break
                    data = json.loads(data_str)
                    content = data["choices"][0]["delta"].get("content", "")
                    print(content, end="", flush=True)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Start the server first in a separate process or assume it is running
    # For verification within this environment, we might need to use a mock or run uvicorn
    test_openai_api()
