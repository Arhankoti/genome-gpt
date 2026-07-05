"""The bridge. A frontier model handles language; it calls the local genome
model for any DNA-native computation. The schema is the boundary: the frontier
model never reasons over raw bases, the genome model never sees prose.

    export ANTHROPIC_API_KEY=...      # or OPENAI_API_KEY
    python bridge/agent.py "Is the mutation at position 50 (C->T) in ATG...GCA disruptive?"

Defaults to Anthropic. Pass --openai to use OpenAI.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bridge.schemas import ANTHROPIC_TOOLS, OPENAI_FUNCTIONS
from bridge.tools import dispatch
from config import Config

SYSTEM = (
    "You are a genomics assistant. You cannot reason over raw DNA bases yourself; "
    "you must call the provided tools for any sequence scoring, generation, variant "
    "effect, or embedding. Extract sequences/positions from the user's request, call "
    "the right tool, then explain the numeric result in plain language. Bits-per-"
    "nucleotide near 2.0 means random; lower means more natural. A negative variant "
    "LLR means the substitution is more disruptive."
)


def run_anthropic(user_msg, model_name):
    import anthropic

    client = anthropic.Anthropic()
    messages = [{"role": "user", "content": user_msg}]
    while True:
        resp = client.messages.create(
            model=model_name,
            max_tokens=1024,
            system=SYSTEM,
            tools=ANTHROPIC_TOOLS,
            messages=messages,
        )
        if resp.stop_reason != "tool_use":
            return "".join(b.text for b in resp.content if b.type == "text")
        messages.append({"role": "assistant", "content": resp.content})
        results = []
        for block in resp.content:
            if block.type == "tool_use":
                out = dispatch(block.name, block.input)
                print(f"  [tool] {block.name}({block.input}) -> {out}")
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(out),
                    }
                )
        messages.append({"role": "user", "content": results})


def run_openai(user_msg, model_name):
    from openai import OpenAI

    client = OpenAI()
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_msg}]
    while True:
        resp = client.chat.completions.create(
            model=model_name,
            messages=messages,
            tools=OPENAI_FUNCTIONS,
        )
        msg = resp.choices[0].message
        if not msg.tool_calls:
            return msg.content
        messages.append(msg)
        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments)
            out = dispatch(tc.function.name, args)
            print(f"  [tool] {tc.function.name}({args}) -> {out}")
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(out)})


def main():
    cfg = Config()
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt")
    ap.add_argument("--openai", action="store_true")
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    if args.openai:
        if not os.environ.get("OPENAI_API_KEY"):
            print("error: OPENAI_API_KEY not set", file=sys.stderr)
            sys.exit(1)
        print(run_openai(args.prompt, args.model or "gpt-4o"))
    else:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("error: ANTHROPIC_API_KEY not set", file=sys.stderr)
            sys.exit(1)
        print(run_anthropic(args.prompt, args.model or cfg.frontier_model))


if __name__ == "__main__":
    main()
