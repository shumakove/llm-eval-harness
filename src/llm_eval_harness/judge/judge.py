import ast
import json
import os

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

DECOMPOSE_SYSTEM_PROMPT_V1 = (
    "Split the given answer into a list of atomic claims. Each claim must be "
    "a single, independently checkable factual statement — do not join two "
    "facts with 'and' or a relative clause. Resolve every pronoun or "
    "reference to the specific entity it refers to (e.g. 'he' -> the "
    "person's name), so each claim can be understood on its own, without the "
    "rest of the answer. Call record_claims with the result."
)

DECOMPOSE_SYSTEM_PROMPT_V2 = (
    "You are given a question and the answer that was produced for it, in "
    "the form 'Question: ...' on one line and 'Answer: ...' on the next. "
    "Split the ANSWER into a list of atomic claims. Each claim must be a "
    "single, independently checkable factual statement — do not join two "
    "facts with 'and' or a relative clause.\n\n"
    "Every claim must stand on its own, without the question and without "
    "the rest of the answer. Resolve every pronoun or reference to the "
    "specific entity it refers to (e.g. 'he' -> the person's name). When "
    "the answer is elliptical — a bare name, phrase or list that only makes "
    "sense as a reply — take the missing wording from the question, so the "
    "fragment becomes a complete sentence.\n\n"
    "Restate only what the answer commits to. The question may supply "
    "wording the answer leaves implicit; it may not add facts the answer "
    "does not assert, and no part of the question becomes a claim of its "
    "own.\n\n"
    "Example — Question: 'Which page is missing from Caplin's census?' "
    "Answer: 'The Veyr family page.' -> a single claim: 'The Veyr family "
    "page is missing from Caplin's census.'\n\n"
    "If the answer asserts nothing — it declines, states that it does not "
    "know, or asks for clarification — return an empty list. Call "
    "record_claims with the result."
)

class JudgeError(RuntimeError): pass

class AtomicClaims(BaseModel):
    claims: list[str]

_DECOMPOSE_TOOL = {
    "type": "function",
    "function": {
        "name": "record_claims",
        "description": "Record the atomic claims extracted from an answer.",
        "parameters": {
            "type": "object",
            "properties": {
                "claims": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["claims"],
        },
    },
}


def _coerce_stringified_literals(payload: dict) -> dict:
    # am/llama-3.2-11b-vision-instruct tends to render composite values as a
    # string instead of a native JSON array/object — sometimes valid JSON
    # re-encoded as a string (double quotes, lowercase true/false), sometimes
    # a Python repr (single quotes, True/False). Try both parsers.
    coerced = {}
    for key, value in payload.items():
        if isinstance(value, str) and value.strip()[:1] in "[{":
            for parse in (json.loads, ast.literal_eval):
                try:
                    value = parse(value)
                    break
                except (ValueError, SyntaxError):
                    continue
        coerced[key] = value
    return coerced


def _parse_tool_call(raw_arguments: str, schema: type[BaseModel]) -> BaseModel:
    # anymodel.org's am/llama-3.2-11b-vision-instruct wraps arguments as
    # {"name": ..., "parameters": {...}} instead of returning the schema
    # fields directly, so unwrap "parameters" when present.
    try:
        data = json.loads(raw_arguments)
        payload = data.get("parameters", data)
        payload = _coerce_stringified_literals(payload)
        return schema.model_validate(payload)
    except ValueError as e:
        raise JudgeError(f"_parse_tool_call failed {e}")

_JUDGE_MODEL = "ag/gemini-3.7-flash-high"


def _build_client() -> OpenAI:
    load_dotenv()
    return OpenAI(
        base_url="https://anymodel.org/v1",
        api_key=os.environ["ANYMODEL_API_KEY"] ,
    )


def decompose_claims(answer: str, *, query:str) -> list[str]:
    client = _build_client()

    response = client.chat.completions.create(
        model=_JUDGE_MODEL,
        messages=[
            {"role": "system", "content": DECOMPOSE_SYSTEM_PROMPT_V2},
            {"role": "user", "content": "Question: " + query + "\nAnswer: " + answer},
        ],
        tools=[_DECOMPOSE_TOOL],
        tool_choice={"type": "function", "function": {"name": "record_claims"}},
    )

    tool_call = response.choices[0].message.tool_calls[0]
    
    result = _parse_tool_call(tool_call.function.arguments, AtomicClaims)
    return result.claims


VERIFY_SYSTEM_PROMPT = (
    "You are given a numbered list of context passages and a numbered list "
    "of claims. For every claim, decide whether it is directly supported by "
    "the context — a claim is supported only if the context states it or "
    "clearly implies it, not if it merely seems plausible. Return one "
    "verdict per claim, in the same order, by calling record_verdicts."
)


class Verdict(BaseModel):
    claim: str
    supported: bool
    reason: str


class Verdicts(BaseModel):
    verdicts: list[Verdict]


_VERIFY_TOOL = {
    "type": "function",
    "function": {
        "name": "record_verdicts",
        "description": "Record one supported/not-supported verdict per claim.",
        "parameters": {
            "type": "object",
            "properties": {
                "verdicts": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "claim": {"type": "string"},
                            "supported": {"type": "boolean"},
                            "reason": {"type": "string"},
                        },
                        "required": ["claim", "supported", "reason"],
                    },
                },
            },
            "required": ["verdicts"],
        },
    },
}


def _build_verify_user_message(claims: list[str], context: list[str]) -> str:
    context_block = "\n".join(f"[{i}] {chunk}" for i, chunk in enumerate(context, start=1))
    claims_block = "\n".join(f"[{i}] {claim}" for i, claim in enumerate(claims, start=1))
    return f"Context:\n{context_block}\n\nClaims:\n{claims_block}"


def verify_claims(claims: list[str], context: list[str]) -> list[Verdict]:
    client = _build_client()

    response = client.chat.completions.create(
        model=_JUDGE_MODEL,
        messages=[
            {"role": "system", "content": VERIFY_SYSTEM_PROMPT},
            {"role": "user", "content": _build_verify_user_message(claims, context)},
        ],
        tools=[_VERIFY_TOOL],
        tool_choice={"type": "function", "function": {"name": "record_verdicts"}},
    )

    tool_call = response.choices[0].message.tool_calls[0]
    result = _parse_tool_call(tool_call.function.arguments, Verdicts)

    if len(result.verdicts) != len(claims):
        raise JudgeError(
            f"Judge returned {len(result.verdicts)} verdicts for {len(claims)} claims"
        )

    return result.verdicts

def describe_judge():
    return {
        "judge_model": _JUDGE_MODEL,
        "decompose_system_prompt": DECOMPOSE_SYSTEM_PROMPT_V2,
        "verify_system_prompt": VERIFY_SYSTEM_PROMPT
    }