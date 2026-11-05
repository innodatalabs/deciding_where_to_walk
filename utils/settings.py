"""Paths, the OpenRouter API key and the Neo4j connection settings, read from the .env file."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SAMPLES = DATA / "samples"        # small question samples that ship with the repository
DOWNLOADS = DATA / "downloads"    # full benchmarks, downloaded from Hugging Face on request
RUNS = ROOT / "benchmark" / "runs"
PUBLISHED = ROOT / "benchmark" / "published_results"  # the organisers' results, same layout as runs/

load_dotenv(ROOT / ".env")


def openrouter_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key or key.startswith("sk-or-..."):
        raise RuntimeError("No OpenRouter key found. Copy .env.example to .env and set OPENROUTER_API_KEY.")
    return key


def neo4j_settings() -> tuple[str, tuple[str, str]]:
    """The Neo4j URI and (user, password). The defaults match .devcontainer/docker-compose.yml."""
    uri = os.environ.get("NEO4J_URI", "bolt://neo4j:7687")
    auth = (os.environ.get("NEO4J_USER", "neo4j"), os.environ.get("NEO4J_PASSWORD", "neo4j-password"))
    return uri, auth
