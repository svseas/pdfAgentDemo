import httpx
import json
import asyncio
from typing import AsyncGenerator, Dict

BASE_URL = "http://127.0.0.1:1234/v1"

async def stream_chat_completion(
    messages: list[Dict[str, str]],
    model: str = "qwen2.5-7b-instruct-1m"
) -> AsyncGenerator[str, None]:
    """
    Stream chat completion from local LM Studio API
    """
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{BASE_URL}/chat/completions",
            json={
                "model": model,
                "messages": messages,
                "stream": True,
            },
            timeout=30.0,
        )
        
        async for line in response.aiter_lines():
            if line.strip():
                if line.startswith("data: "):
                    json_str = line[6:]  # Remove "data: " prefix
                    if json_str.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(json_str)
                        if content := chunk["choices"][0]["delta"].get("content"):
                            yield content
                    except json.JSONDecodeError:
                        print(f"Failed to parse JSON: {json_str}")
                        continue

async def main():
    messages = [
        {"role": "user", "content": "Write a short poem about streaming AI responses"}
    ]
    
    print("\nStarting stream (each | represents a token):\n")
    print("Response: ", end="", flush=True)
    
    async for token in stream_chat_completion(messages):
        print(f"{token}|", end="", flush=True)
        await asyncio.sleep(0.1)  # Add delay to make streaming more visible
    
    print("\n\nStream completed!")

if __name__ == "__main__":
    asyncio.run(main())