"""
{
    "openai_config": {
        "api_key": "your-api-key-here",
        "api_base": "https://api.openai.com/v1"
    }
}
"""

import json
import openai
from openai import OpenAI
from pathlib import Path

def load_config(config_path="config.json"):
    """Load API configuration from JSON file"""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
            return config['openai_config']
    except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
        print(f"Error loading config: {e}")
        exit(1)

def initialize_openai_client(config):
    """Initialize OpenAI client with configuration"""
    return OpenAI(
        api_key=config['api_key'],
        base_url=config.get('api_base', 'https://api.openai.com/v1')
    )

def get_chat_completion(client, prompt, model="gpt-3.5-turbo", temperature=0.7):
    """Get completion from OpenAI API"""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"API Error: {e}")
        return None

def main():
    # Load configuration
    config = load_config()
    
    # Initialize client
    client = initialize_openai_client(config)
    
    # Interactive prompt loop
    print("Chat with AI (type 'exit' to quit)")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ('exit', 'quit'):
            break
        
        # Get and display response
        response = get_chat_completion(client, user_input)
        if response:
            print(f"\nAI: {response}")

if __name__ == "__main__":
    main()
