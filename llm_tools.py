from __future__ import annotations

import os
import json
from typing import Any, Generator
from dataclasses import dataclass, field
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

SYSTEM_PROMPT = """\
You are an expert robotic arm controller assistant. You operate a RoArm robotic arm with a gripper, \
using a vision-language model (VLM) to detect objects in the camera feed.

Your capabilities:
1. **Pick and place objects** — You set VLM grasp targets and place targets, then start execution.
2. **Reset the arm** — Return the arm to its home position.

When the user asks you to pick an object and place it somewhere, you MUST:
1. First call `set_vlm_targets` to set what to grasp and its best anchor/grasp point.
2. Then call `set_vlm_place_targets` to set where to place it and anchor point.
3. Finally call `start_execution` to begin the pick-and-place cycle.

**Multi-task / sequential requests:**
The arm can only hold ONE object at a time. When the user mentions multiple objects, \
you MUST expand it into sequential individual pick-and-place tasks.
Examples:
- "pick the cube, marker, and screwdriver and place onto the tape" → 3 tasks: \
cube→tape, marker→tape, screwdriver→tape.
- "pick the cube and place on the tape, then the wheel again" → 2 tasks: \
cube→tape, wheel→tape.
- "move all items to the box" → one task per item, all placed on the box.

Rules:
- Plan ALL tasks from the request upfront.
- Execute them ONE AT A TIME — call set_vlm_targets, set_vlm_place_targets, start_execution for the FIRST task only.
- After each task completes (you will receive a completion message), \
IMMEDIATELY proceed to the NEXT task by calling the tools again. Do NOT wait for user input.
- The user may also say "again" or "repeat" — this means repeat the same task sequence.
- Continue until ALL tasks are done, then report final summary.

**Anchor point selection rules** (pick the point where the gripper can clamp the object stably):
Available anchors: center, narrow, handle, head, left, right, top, bottom, top_left, top_right, bottom_left, bottom_right.

Special smart anchors (uses mask shape analysis):
- **"handle"** — finds the NARROWER/thinner end of the object. For tools with a wide head.
- **"head"** — finds the WIDER/bulkier end. Opposite of handle.
- **"narrow"** — finds the thinnest width cross-section of the mask. Only use for special cases \
where you need to grip at the exact thinnest point (e.g. hourglass shapes, tapered objects).

Object rules:
- **cube / box / block / dice / ball / sphere**: "center" — symmetric shapes.
- **bottle / cylinder / can**: "center" — cylindrical symmetry.
- **coin / card**: "center" — flat symmetric objects.
- **cup / mug**: "center" — grip the body.
- **tape / roll / wheel / tire / donut / ring** (objects with a hole): \
MUST use "left" or "right" when PICKING — NEVER "center" because the hole is there and the gripper will miss. \
Only use "center" for these when they are a PLACE destination (set_vlm_place_targets).
- **brush / toothbrush**: "narrow" — has bristles the gripper can't pick, grip the thinnest part of the handle.
- **marker / pen / pencil / crayon / screwdriver**: "center" — uniform handles, grip the body center.
- **spoon / fork / wrench / hammer / spatula**: "handle" — \
grip the narrow handle end, away from the wide functional head.
- **knife**: "handle" — grip the handle, not the blade.
- **irregular / elongated / unknown tool-like shapes**: use "center" to grip the body. \
Only use "handle" if you specifically need the narrow end (e.g. a tool with a wide head).

For **set_vlm_place_targets** (place destinations), always use "center" unless the user specifies otherwise.

If the user just says "reset", "go home", or "stop", call `reset_to_home`.

After calling `start_execution`, you MUST tell the user what you are doing and ask them to wait. \
For example: "I'm now executing the pick-and-place. The arm is reaching for the [object]... please wait."

When you receive execution results (success, failure, or stopped), respond naturally:
- On success: if there are remaining tasks, immediately proceed with the next one. \
If all tasks are done, confirm everything completed.
- On failure: explain what went wrong and suggest next steps.
- On stopped: acknowledge the user stopped it and do NOT continue remaining tasks.

Always explain briefly what you are doing before calling tools. Be concise.\
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "set_vlm_targets",
            "description": (
                "Set the VLM grasp target — the SINGLE object the arm should pick up in this cycle. "
                "Only set ONE object per execution cycle (the arm can only hold one object at a time). "
                "Anchor can be: center, narrow, handle, head, left, right, top, bottom, top_left, top_right, bottom_left, bottom_right. "
                "CRITICAL: For tape/roll/wheel/tire/ring (objects with a HOLE), MUST use 'left' or 'right' — "
                "NEVER 'center' because the gripper will miss the hole. "
                "Use 'center' for solid objects. Use 'handle' for tools with a wide head. "
                "Use 'narrow' for brush/toothbrush."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "targets": {
                        "type": "object",
                        "description": "ONE object mapped to anchor. e.g. {\"cube\": \"center\"}, {\"tape\": \"left\"}, {\"brush\": \"narrow\"}",
                        "additionalProperties": {"type": "string"},
                    }
                },
                "required": ["targets"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_vlm_place_targets",
            "description": (
                "Set the VLM place target — the SINGLE destination where the grasped object should be placed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "targets": {
                        "type": "object",
                        "description": "ONE place target name mapped to its anchor point, e.g. {\"tape\": \"center\"}",
                        "additionalProperties": {"type": "string"},
                    }
                },
                "required": ["targets"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "start_execution",
            "description": (
                "Start the pick-and-place execution cycle. "
                "Call this AFTER setting both vlm_targets and vlm_place_targets. "
                "The arm will reach for the object, grasp it, lift, move to the place target, and release. "
                "After calling this, tell the user you are executing and to please wait."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "reset_to_home",
            "description": (
                "Reset the robotic arm to its home position. "
                "Use when the user says reset, stop, go home, or abort."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_execution_status",
            "description": (
                "Get the current execution phase/status of the robotic arm. "
                "Returns the current phase (idle, reaching, grasping, lifting, "
                "lift_settle, placing, place_settle) and whether execution is active. "
                "Use this to check if the arm is busy or idle."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
]


@dataclass
class ChatMessage:
    role: str
    content: str
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None
    name: str | None = None


@dataclass
class LLMClient:
    model: str = ""
    messages: list[dict[str, Any]] = field(default_factory=list)
    _client: OpenAI | None = None

    def __post_init__(self):
        base_url = os.environ.get(
            "OPENAI_BASE_URL", "http://localhost:11434/v1")
        api_key = os.environ.get("OPENAI_API_KEY", "ollama")

        self._client = OpenAI(base_url=base_url, api_key=api_key)

        if not self.model:
            env_model = os.environ.get("OPENAI_MODEL", "")
            if env_model:
                self.model = env_model
            else:
                models = self.list_models()
                self.model = models[0] if models else ""

        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    def list_models(self) -> list[str]:
        """Query the API for available models."""
        try:
            models = self._client.models.list()
            return sorted([m.id for m in models.data])
        except Exception:
            return []

    def reset_history(self):
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    def get_history(self) -> list[dict[str, Any]]:
        """Return non-system messages for display."""
        return [m for m in self.messages if m["role"] != "system"]

    def send_message(self, user_text: str) -> Generator[dict[str, Any], None, None]:
        """
        Send user message and yield events:
          {"type": "text_delta", "content": "..."}
          {"type": "tool_call", "name": "...", "arguments": {...}}
          {"type": "done"}
          {"type": "error", "content": "..."}
        """
        self.messages.append({"role": "user", "content": user_text})

        try:
            yield from self._run_completion()
        except Exception as e:
            yield {"type": "error", "content": str(e)}

    def _run_completion(self) -> Generator[dict[str, Any], None, None]:
        """Run one completion, handle tool calls, loop if needed."""
        while True:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                tools=TOOLS,
                tool_choice="auto",
                stream=True,
            )

            collected_content = ""
            tool_calls_acc: dict[int, dict] = {}

            for chunk in response:
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta is None:
                    continue

                if delta.content:
                    collected_content += delta.content
                    yield {"type": "text_delta", "content": delta.content}

                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = tc.index
                        if idx not in tool_calls_acc:
                            tool_calls_acc[idx] = {
                                "id": tc.id or "",
                                "name": tc.function.name or "" if tc.function else "",
                                "arguments": "",
                            }
                        if tc.id:
                            tool_calls_acc[idx]["id"] = tc.id
                        if tc.function:
                            if tc.function.name:
                                tool_calls_acc[idx]["name"] = tc.function.name
                            if tc.function.arguments:
                                tool_calls_acc[idx]["arguments"] += tc.function.arguments

            assistant_msg: dict[str, Any] = {"role": "assistant"}
            if collected_content:
                assistant_msg["content"] = collected_content
            else:
                assistant_msg["content"] = None

            if tool_calls_acc:
                assistant_msg["tool_calls"] = []
                for idx in sorted(tool_calls_acc.keys()):
                    tc = tool_calls_acc[idx]
                    assistant_msg["tool_calls"].append({
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": tc["arguments"],
                        },
                    })

            self.messages.append(assistant_msg)

            if not tool_calls_acc:
                yield {"type": "done"}
                return

            for idx in sorted(tool_calls_acc.keys()):
                tc = tool_calls_acc[idx]

                try:
                    args = json.loads(
                        tc["arguments"]) if tc["arguments"] else {}
                except json.JSONDecodeError:
                    args = {}

                yield {
                    "type": "tool_call",
                    "name": tc["name"],
                    "arguments": args,
                    "tool_call_id": tc["id"],
                }

                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": "__PENDING__",
                })

            yield {"type": "awaiting_tool_results"}
            return

    def inject_tool_result(self, tool_call_id: str, result: str):
        """Replace the pending tool result with the actual result."""
        for msg in reversed(self.messages):
            if (msg.get("role") == "tool"
                    and msg.get("tool_call_id") == tool_call_id
                    and msg.get("content") == "__PENDING__"):
                msg["content"] = result
                return

    def continue_after_tools(self) -> Generator[dict[str, Any], None, None]:
        """Continue the conversation after tool results are injected."""
        try:
            yield from self._run_completion()
        except Exception as e:
            yield {"type": "error", "content": str(e)}
