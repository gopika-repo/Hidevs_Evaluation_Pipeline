"""
Ragas compatibility patch for LangChain restructure.

WHY this patch exists:
Ragas 0.4.3's internal code imports `langchain_community.chat_models.vertexai`,
which no longer exists in current LangChain versions since VertexAI integrations
moved to `langchain-google-vertexai`.

WHAT it does:
Creates a placeholder module `langchain_community.chat_models.vertexai` with a
dummy `ChatVertexAI` class so Ragas's import succeeds. This project doesn't actually
use VertexAI (it uses Gemini directly via langchain-google-genai).

TODO:
Check if a newer Ragas release has fixed this import path before removing this patch.
If found, remove this file and re-test Retrieval Evaluator.
"""

import sys
import types

if "langchain_community.chat_models.vertexai" not in sys.modules:
    m = types.ModuleType("langchain_community.chat_models.vertexai")
    sys.modules["langchain_community.chat_models.vertexai"] = m
    m.ChatVertexAI = object
