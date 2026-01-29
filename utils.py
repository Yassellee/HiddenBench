from typing import List, Dict
from openai import OpenAI
from dotenv import load_dotenv
import json, os, time
from pydantic import BaseModel, Field

load_dotenv()

POSSIBLE_MODELS = [
    'gpt-4o', 
    'gemini-2.5-flash', 
    'meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8', 
    'gpt-4.1', 
    'gpt-4.1-mini', 
    'gpt-4.1-nano', 
    'gpt-5-medium', 
    'qwen3-32b', 
    'gpt-5-minimal', 
    'gpt-5-mini-minimal',
    'gpt-5-nano-minimal',
    'gpt-4.1-mini',
    'gpt-4.1-nano',
    'gemini-2.5-flash-lite',
    'gemini-2.5-pro',
    'qwen3-235b-a22b',
    'qwen3-14b',
    'qwen3-8b',
    'meta-llama/Llama-4-Scout-17B-16E-Instruct'
]

client_openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
client_google = OpenAI(api_key=os.getenv("GOOGLE_API_KEY"), base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
client_anthropic = OpenAI(api_key=os.getenv("ANTHROPIC_API_KEY"), base_url="https://api.anthropic.com/v1/")
client_together = OpenAI(api_key=os.getenv("TOGETHER_API_KEY"), base_url="https://api.together.xyz/v1")
client_qwen = OpenAI(api_key=os.getenv("DASHSCOPE_API_KEY"), base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
client_openrouter = OpenAI(api_key=os.getenv("OPENROUTER_API_KEY"), base_url="https://openrouter.ai/api/v1")

def replace_prompt(raw_prompt: str, information: Dict[str, str]) -> str:
    for key, value in information.items():
        raw_prompt = raw_prompt.replace(f"%{key}%", value)

    return raw_prompt

def json_chat(messages: List[Dict[str, str]],
              response_format,
              model: str) -> str:
    def extract_json(input_string):
        input_string = input_string.replace("\n", "")
        stack = []
        json_start_positions = []

        for pos, char in enumerate(input_string):
            if char in '{[':
                stack.append(char)
                if len(stack) == 1:
                    json_start_positions.append(pos)
            elif char in '}]':
                if len(stack) == 0:
                    raise ValueError(f"unexpected {char} at position {pos}")
                last_open = stack.pop()
                if (last_open == '{' and char != '}') or (last_open == '[' and char != ']'):
                    raise ValueError(f"mismatched brackets {last_open} and {char} at position {pos}")
                if len(stack) == 0:
                    return input_string[json_start_positions.pop():pos+1]
        return None

    def openai_json_chat():
        while True:
            try:
                response = client_openai.beta.chat.completions.parse(
                    model=model,
                    messages=messages,
                    response_format=response_format,
                )
                result = response.choices[0].message.content
                
                json_content = extract_json(result)
                if json_content:
                    result = json.loads(json_content)
                else:
                    raise ValueError("No JSON content found")
                
                return result
            
            except Exception as e:
                print(f"Error in openai_json_chat: {e}")
                continue

    def gpt_5_minimal_json_chat():
        while True:
            try:
                response = client_openai.beta.chat.completions.parse(
                    model='gpt-5',
                    messages=messages,
                    response_format=response_format,
                    reasoning_effort="minimal"
                )
                result = response.choices[0].message.content
                
                json_content = extract_json(result)
                if json_content:
                    result = json.loads(json_content)
                else:
                    raise ValueError("No JSON content found")
                
                return result
            
            except Exception as e:
                print(f"Error in gpt_5_minimal_json_chat: {e}")
                continue

    def gpt_5_medium_json_chat():
        while True:
            try:
                response = client_openai.beta.chat.completions.parse(
                    model='gpt-5',
                    messages=messages,
                    response_format=response_format,
                    reasoning_effort="medium"
                )
                result = response.choices[0].message.content
                
                json_content = extract_json(result)
                if json_content:
                    result = json.loads(json_content)
                else:
                    raise ValueError("No JSON content found")
                
                return result
            
            except Exception as e:
                print(f"Error in gpt_5_medium_json_chat: {e}")
                continue

    def gpt_5_mini_minimal_json_chat():
        while True:
            try:
                response = client_openai.beta.chat.completions.parse(
                    model='gpt-5-mini',
                    messages=messages,
                    response_format=response_format,
                    reasoning_effort="minimal"
                )
                result = response.choices[0].message.content
                
                json_content = extract_json(result)
                if json_content:
                    result = json.loads(json_content)
                else:
                    raise ValueError("No JSON content found")
                
                return result
            
            except Exception as e:
                print(f"Error in gpt_5_mini_minimal_json_chat: {e}")
                continue

    def gpt_5_nano_minimal_json_chat():
        while True:
            try:
                response = client_openai.beta.chat.completions.parse(
                    model='gpt-5-nano',
                    messages=messages,
                    response_format=response_format,
                    reasoning_effort="minimal"
                )
                result = response.choices[0].message.content
                
                json_content = extract_json(result)
                if json_content:
                    result = json.loads(json_content)
                else:
                    raise ValueError("No JSON content found")
                
                return result
            
            except Exception as e:
                print(f"Error in gpt_5_nano_minimal_json_chat: {e}")
                continue
            
    def google_json_chat():
        while True:
            try:
                response = client_google.beta.chat.completions.parse(
                    model=model,
                    messages=messages,
                    response_format=response_format
                )
                result = response.choices[0].message.content
                
                json_content = extract_json(result)
                if json_content:
                    result = json.loads(json_content)
                else:
                    raise ValueError("No JSON content found")
                
                return result
            
            except Exception as e:
                print(f"Error in google_json_chat: {e}")
                continue

    def anthropic_json_chat():
        while True:
            try:
                response = client_anthropic.chat.completions.create(
                    model=model,
                    messages=messages,
                )
                result = response.choices[0].message.content
                
                json_content = extract_json(result)
                if json_content:
                    result = json.loads(json_content)
                else:
                    raise ValueError("No JSON content found")
                
                return result
            
            except Exception as e:
                print(f"Error in anthropic_json_chat: {e}")
                continue

    def together_json_chat():
        while True:
            try:
                response = client_together.beta.chat.completions.parse(
                    model=model,
                    messages=messages,
                    response_format={
                        "type": "json_schema",
                        "schema": response_format.model_json_schema()
                    }
                )
                result = response.choices[0].message.content
                
                json_content = extract_json(result)
                if json_content:
                    result = json.loads(json_content)
                else:
                    raise ValueError("No JSON content found")
                
                return result
            
            except Exception as e:
                print(f"Error in together_json_chat: {e}")
                continue

    def qwen_json_chat():
        while True:
            try:
                response = client_qwen.chat.completions.create(
                    model=model,
                    messages=messages,
                    response_format={"type": "json_object"},
                    extra_body={"enable_thinking": False}
                )
                result = response.choices[0].message.content
                
                json_content = extract_json(result)
                if json_content:
                    result = json.loads(json_content)
                else:
                    raise ValueError("No JSON content found")
                
                return result
            
            except Exception as e:
                print(f"Error in qwen_json_chat: {e}")
                time.sleep(3)
                continue

    def openrouter_json_chat():
        while True:
            try:
                response = client_openrouter.chat.completions.create(
                    model='google/gemini-2.5-flash',
                    messages=messages,
                )
                result = response.choices[0].message.content
                
                json_content = extract_json(result)
                if json_content:
                    result = json.loads(json_content)
                else:
                    raise ValueError("No JSON content found")
            except Exception as e:
                print(f"Error in openrouter_json_chat: {e}")
                time.sleep(3)
                continue

    if model in POSSIBLE_MODELS:
        if model == 'gpt-4o' or model == 'gpt-4.1' or model == 'gpt-4.1-mini' or model == 'gpt-4.1-nano':
            return openai_json_chat()
        elif model == 'gpt-5-medium':
            return gpt_5_medium_json_chat()
        elif model == 'gpt-5-minimal':
            return gpt_5_minimal_json_chat()
        elif model == 'gemini-2.5-flash' or model == 'gemini-2.5-flash-lite' or model == 'gemini-2.5-pro':
            return google_json_chat()
        elif model == 'meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8' or model == 'meta-llama/Llama-4-Scout-17B-16E-Instruct':
            return together_json_chat()
        elif model == 'qwen3-32b' or model == 'qwen3-235b-a22b' or model == 'qwen3-14b' or model == 'qwen3-8b':
            return qwen_json_chat()
        elif model == 'gpt-5-mini-minimal':
            return gpt_5_mini_minimal_json_chat()
        elif model == 'gpt-5-nano-minimal':
            return gpt_5_nano_minimal_json_chat()
    else:
        raise ValueError(f"Model {model} not supported")


def normal_chat(messages: List[Dict[str, str]], model: str) -> str:
    
    def openai_normal_chat():
        while True:
            try:
                response = client_openai.chat.completions.create(
                    model=model,
                    messages=messages,
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"Error in openai_normal_chat: {e}")
                continue

    def gpt_5_minimal_normal_chat():
        while True:
            try:
                response = client_openai.chat.completions.create(
                    model='gpt-5',
                    messages=messages,
                    reasoning_effort="minimal"
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"Error in gpt_5_minimal_normal_chat: {e}")
                continue

    def gpt_5_mini_minimal_normal_chat():
        while True:
            try:
                response = client_openai.chat.completions.create(
                    model='gpt-5-mini',
                    messages=messages,
                    reasoning_effort="minimal"
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"Error in gpt_5_mini_minimal_normal_chat: {e}")
                continue

    def gpt_5_medium_normal_chat():
        while True:
            try:
                response = client_openai.chat.completions.create(
                    model='gpt-5',
                    messages=messages,
                    reasoning_effort="medium"
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"Error in gpt_5_medium_normal_chat: {e}")
                continue

    def gpt_5_nano_minimal_normal_chat():
        while True:
            try:
                response = client_openai.chat.completions.create(
                    model='gpt-5-nano',
                    messages=messages,
                    reasoning_effort="minimal"
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"Error in gpt_5_nano_minimal_normal_chat: {e}")
                continue

    def anthropic_normal_chat():
        while True:
            try:
                response = client_anthropic.chat.completions.create(
                    model=model,
                    messages=messages,
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"Error in anthropic_normal_chat: {e}")   
                continue
    
    def google_normal_chat():
        while True:
            try:
                response = client_google.chat.completions.create(
                    model=model,
                    messages=messages,
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"Error in google_normal_chat: {e}")
                continue

    def together_normal_chat():
        while True:
            try:
                response = client_together.chat.completions.create(
                    model=model,
                    messages=messages,
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"Error in together_normal_chat: {e}")
                continue

    def qwen_normal_chat():
        while True:
            try:
                response = client_qwen.chat.completions.create(
                    model=model,
                    messages=messages,
                    response_format={"type": "text"},
                    extra_body={"enable_thinking": False}
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"Error in qwen_normal_chat: {e}")
                continue

    def openrouter_normal_chat():
        while True:
            try:
                response = client_openrouter.chat.completions.create(
                    model='google/gemini-2.5-flash',
                    messages=messages,
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"Error in openrouter_normal_chat: {e}")
                continue
    
    if model in POSSIBLE_MODELS:
        if model == 'gpt-4o' or model == 'gpt-4.1' or model == 'gpt-4.1-mini' or model == 'gpt-4.1-nano':
            return openai_normal_chat()
        elif model == 'gpt-5-medium':
            return gpt_5_medium_normal_chat()
        elif model == 'gpt-5-minimal':
            return gpt_5_minimal_normal_chat()
        elif model == 'gpt-5-mini-minimal':
            return gpt_5_mini_minimal_normal_chat()
        elif model == 'gpt-5-nano-minimal':
            return gpt_5_nano_minimal_normal_chat()
        elif model == 'gemini-2.5-flash' or model == 'gemini-2.5-flash-lite' or model == 'gemini-2.5-pro':
            return google_normal_chat()
        elif model == 'meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8' or model == 'meta-llama/Llama-4-Scout-17B-16E-Instruct':
            return together_normal_chat()
        elif model == 'qwen3-32b' or model == 'qwen3-235b-a22b' or model == 'qwen3-14b' or model == 'qwen3-8b':
            return qwen_normal_chat()
    else:
        raise ValueError(f"Model {model} not supported")


class VoteResponse(BaseModel):
    vote: str
    rationale: str

class DecisionResponse(BaseModel):
    decision: str
    rationale: str

class TaskResponse(BaseModel):
    name: str
    description: str
    shared_information: List[str]
    hidden_information: List[str]
    possible_answers: List[str]
    correct_answer: str