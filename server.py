#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
DeepSeek API Server - OpenAI-compatible server for DeepSeek
'''

import os
import json
import time
import sys
from io import StringIO
from flask import Flask, request, jsonify, Response, stream_with_context
from DeepSeekAPI import DeepSeekChat
from tools import list_tools, execute_tool

app = Flask(__name__)

# Load tokens from file or environment
def get_tokens():
    if os.path.exists("tokens"):
        with open("tokens") as f:
            lines = f.read().strip().split('\n')
            if len(lines) >= 2:
                return lines[0], lines[1]
    ds_session_id = os.environ.get("DS_SESSION_ID")
    authorization_token = os.environ.get("AUTHORIZATION_TOKEN")
    if ds_session_id and authorization_token:
        return ds_session_id, authorization_token
    raise ValueError("Tokens not found. Set DS_SESSION_ID and AUTHORIZATION_TOKEN or create 'tokens' file.")

DS_SESSION_ID, AUTHORIZATION_TOKEN = get_tokens()

def get_model_config(model: str):
    """Map OpenAI-compatible model names to DeepSeek web model flags."""
    model_lower = model.lower()
    model_type = "expert" if "v4" in model_lower or "r4" in model_lower or "expert" in model_lower else "default"
    thinking_enabled = "r1" in model_lower or "r4" in model_lower or "reasoning" in model_lower or "reasoner" in model_lower
    return model_type, thinking_enabled


def build_tool_prompt(messages, tools):
    """Convert OpenAI messages/tools into a prompt DeepSeek can understand."""

    conversation = []

    for message in messages:
        role = message.get("role", "user")
        content = message.get("content") or ""

        # Preserve assistant tool-call context.
        if role == "assistant" and message.get("tool_calls"):
            conversation.append(
                "ASSISTANT TOOL CALLS:\n" +
                json.dumps(message["tool_calls"], ensure_ascii=False)
            )
            continue

        # Preserve tool results returned by the client.
        if role == "tool":
            tool_call_id = message.get("tool_call_id", "")
            name = message.get("name", "")
            conversation.append(
                f"TOOL RESULT"
                f"{' ' + name if name else ''}"
                f"{' (' + tool_call_id + ')' if tool_call_id else ''}:\n"
                f"{content}"
            )
            continue

        conversation.append(f"{role.upper()}:\n{content}")

    tool_defs = []

    for tool in tools or []:
        if tool.get("type") != "function":
            continue

        fn = tool.get("function", {})

        tool_defs.append({
            "name": fn.get("name"),
            "description": fn.get("description", ""),
            "parameters": fn.get("parameters", {
                "type": "object",
                "properties": {}
            })
        })

    if not tool_defs:
        return "\n\n".join(conversation)

    instructions = """
You have access to tools.

If a tool is required, respond ONLY with valid JSON in exactly this form:

{"tool_call":{"name":"TOOL_NAME","arguments":{}}}

Do not wrap the JSON in markdown.
Do not explain the tool call.
Use only tools listed below.

If no tool is required, answer normally.

AVAILABLE TOOLS:
""" + json.dumps(tool_defs, ensure_ascii=False, indent=2)

    return instructions + "\n\nCONVERSATION:\n" + "\n\n".join(conversation)


def parse_tool_call(text):
    """Parse the gateway's JSON tool-call protocol."""

    if not isinstance(text, str):
        return None

    candidate = text.strip()

    # Tolerate models wrapping JSON in a markdown fence.
    if candidate.startswith("```"):
        lines = candidate.splitlines()

        if lines and lines[0].startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        candidate = "\n".join(lines).strip()

    try:
        data = json.loads(candidate)
    except (json.JSONDecodeError, TypeError):
        return None

    call = data.get("tool_call")

    if not isinstance(call, dict):
        return None

    name = call.get("name")
    arguments = call.get("arguments", {})

    if not isinstance(name, str) or not name:
        return None

    if not isinstance(arguments, dict):
        return None

    return {
        "name": name,
        "arguments": arguments
    }


def chat_non_streaming(messages, model_type="default", thinking_enabled=True):
    """Non-streaming chat"""

    user_message = messages[-1]["content"] if messages else ""

    chat = DeepSeekChat(DS_SESSION_ID, AUTHORIZATION_TOKEN)
    chat.chat_session_id = None
    chat.parent_message_id = None

    result = chat.send_message(
        user_message,
        printing=False,
        thinking_enabled=thinking_enabled,
        search_enabled=False,
        model_type=model_type
    )

    if result and result.get("ok"):
        response = result["content"].get("response", "")

        # Remove leaked reasoning prefixes
        markers = [
            "We need answer",
            "Need answer",
            "Let's craft",
            "Need to",
            "User asks"
        ]

        for marker in markers:
            if response.startswith(marker):
                idx = response.find("\\n\\n")
                if idx != -1:
                    response = response[idx+2:]

        return response

    return ""


def chat_streaming(messages, model_type="default", thinking_enabled=True):
    """Streaming chat using SSE"""

    user_message = messages[-1]["content"] if messages else ""

    chat = DeepSeekChat(DS_SESSION_ID, AUTHORIZATION_TOKEN)
    chat.chat_session_id = None
    chat.parent_message_id = None

    result = chat.send_message(
        user_message,
        printing=False,
        thinking_enabled=thinking_enabled,
        search_enabled=False,
        model_type=model_type
    )

    if result and result.get("ok"):
        response_text = result["content"].get("response", "")
    else:
        response_text = ""

    # Emit OpenAI-compatible SSE chunks
    chunk_size = 20

    for i in range(0, len(response_text), chunk_size):
        content = response_text[i:i+chunk_size]

        data_str = json.dumps({
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "content": content
                    }
                }
            ]
        })

        yield "data: " + data_str + "\n\n"

    yield "data: [DONE]\n\n"


@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    data = request.json
    
    messages = data.get("messages", [])

    # Remove only failed OpenClaw assistant turns
    bad_phrases = [
        "[assistant turn failed before producing content]",
        "don't have a previous question or task to continue from",
        "don't have the earlier context from this conversation",
        "don't have access to the previous conversation state",
        "Could you tell me what you'd like me to answer",
        "Could you share the previous question or context",
    ]

    messages = [
        m for m in messages
        if not (
            m.get("role") == "assistant"
            and any(
                phrase in m.get("content", "")
                for phrase in bad_phrases
            )
        )
    ]

    stream = data.get("stream", False)
    tools = data.get("tools", [])
    
    # Determine model - default to DeepSeek V3.
    model = data.get("model", "deepseek-v3")
    
    # Determine DeepSeek web model flags based on model name.
    model_type, thinking_enabled = get_model_config(model)
    
    # OpenAI-compatible tool calling.
    # For now this path is non-streaming; streaming tool calls are added separately.
    if tools and not stream:
        tool_prompt = build_tool_prompt(messages, tools)

        result = chat_non_streaming(
            [{"role": "user", "content": tool_prompt}],
            model_type,
            thinking_enabled
        )

        tool_call = parse_tool_call(result)

        if tool_call:
            # Only allow tools actually advertised by the client.
            allowed_tools = {
                tool.get("function", {}).get("name")
                for tool in tools
                if tool.get("type") == "function"
            }

            if tool_call["name"] in allowed_tools:
                call_id = f"call_{int(time.time() * 1000000)}"

                return jsonify({
                    "id": f"chatcmpl-{int(time.time())}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [{
                                "id": call_id,
                                "type": "function",
                                "function": {
                                    "name": tool_call["name"],
                                    "arguments": json.dumps(
                                        tool_call["arguments"],
                                        ensure_ascii=False
                                    )
                                }
                            }]
                        },
                        "finish_reason": "tool_calls"
                    }],
                    "usage": {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0
                    }
                })

        # DeepSeek decided no tool was necessary.
        return jsonify({
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": result
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": len(result.split()) if result else 0,
                "total_tokens": len(result.split()) if result else 0
            }
        })

    if stream:
        return Response(
            stream_with_context(chat_streaming(messages, model_type, thinking_enabled)),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
            }
        )
    else:
        result = chat_non_streaming(messages, model_type, thinking_enabled)
        
        return jsonify({
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": result
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": len(result.split()) if result else 0,
                "total_tokens": len(result.split()) if result else 0
            }
        })

@app.route("/v1/models", methods=["GET"])
def list_models():
    return jsonify({
        "object": "list",
        "data": [
            {
                "id": "deepseek-v3",
                "object": "model",
                "created": 1704067200,
                "owned_by": "deepseek",
                "description": "DeepSeek V3 - Fast responses without extended thinking"
            },
            {
                "id": "deepseek-r1",
                "object": "model",
                "created": 1704067200,
                "owned_by": "deepseek",
                "description": "DeepSeek R1 - Reasoning model with extended thinking"
            },
            {
                "id": "deepseek-v4",
                "object": "model",
                "created": 1704067200,
                "owned_by": "deepseek",
                "description": "DeepSeek V4 - Expert model without extended thinking"
            },
            {
                "id": "deepseek-r4",
                "object": "model",
                "created": 1704067200,
                "owned_by": "deepseek",
                "description": "DeepSeek R4 - Expert reasoning model with extended thinking"
            }
        ]
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="DeepSeek API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    args = parser.parse_args()
    
    print(f"Starting DeepSeek API Server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port)
