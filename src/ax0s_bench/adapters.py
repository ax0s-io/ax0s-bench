from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
import asyncio
import copy
import hashlib
import json
import logging
import os
import shutil
import time
import uuid

from .api import OpenAIChatClient


class MemoryAdapter(ABC):
    """Pluggable condition boundary for full-context, AX0S, Mem0, Zep, etc."""

    @abstractmethod
    def ingest_conversation(self, conversation: dict[str, Any]) -> None:
        raise NotImplementedError

    @abstractmethod
    def answer(self, question: str, *, question_index: int) -> dict[str, Any]:
        raise NotImplementedError

    def receipt_metadata(self) -> dict[str, Any]:
        return {}


def render_conversation(conversation: dict[str, Any]) -> str:
    session_numbers = sorted(
        int(key.removeprefix("session_"))
        for key, value in conversation.items()
        if key.startswith("session_") and key.removeprefix("session_").isdigit() and isinstance(value, list)
    )
    blocks: list[str] = []
    for number in session_numbers:
        date = conversation.get(f"session_{number}_date_time", "date unavailable")
        lines = [f"[Session {number} — {date}]"]
        for turn in conversation[f"session_{number}"]:
            lines.append(f'{turn["speaker"]}: {turn["text"]}')
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def parse_locomo_datetime(value: str) -> datetime | None:
    """Parse a LoCoMo session header timestamp, e.g. "1:56 pm on 8 May, 2023"."""
    for fmt in ("%I:%M %p on %d %B, %Y", "%I:%M %p on %d %b, %Y"):
        try:
            return datetime.strptime(value, fmt).replace(tzinfo=timezone.utc)
        except (TypeError, ValueError):
            pass
    return None


def locomo_turn_text(turn: dict[str, Any]) -> str:
    """Render one LoCoMo turn, folding shared-image captions into the text."""
    text = str(turn.get("text", ""))
    blip = str(turn.get("blip_caption", ""))
    query = str(turn.get("query", ""))
    if query and blip:
        tag = f"[Sharing image - query: {query}. The image shows: {blip}]"
    elif query:
        tag = f"[Sharing image - query for: {query}]"
    elif blip:
        tag = f"[Sharing image that shows: {blip}]"
    else:
        tag = ""
    if tag:
        text = f"{text} {tag}" if text else tag
    return text


def chronological_sessions(
    conversation: dict[str, Any], max_sessions: int | None = None
) -> list[tuple[int, str, list[dict[str, Any]]]]:
    """Return (number, date string, turns) sorted by session date, undated last."""
    sessions: list[tuple[int, str, list[dict[str, Any]]]] = []
    for key, turns in conversation.items():
        suffix = key.removeprefix("session_")
        if key.startswith("session_") and suffix.isdigit() and isinstance(turns, list):
            number = int(suffix)
            sessions.append((number, str(conversation.get(f"session_{number}_date_time", "")), turns))
    sessions.sort(key=lambda item: (parse_locomo_datetime(item[1]) is None, parse_locomo_datetime(item[1]) or item[0]))
    return sessions if max_sessions is None else sessions[:max_sessions]


class FullContextAdapter(MemoryAdapter):
    def __init__(self, client: OpenAIChatClient):
        self.client = client
        self.context = ""

    def ingest_conversation(self, conversation: dict[str, Any]) -> None:
        self.context = render_conversation(conversation)

    def answer(self, question: str, *, question_index: int) -> dict[str, Any]:
        if not self.context:
            raise RuntimeError("ingest_conversation must be called before answer")
        return self.client.chat(
            [
                {
                    "role": "system",
                    "content": (
                        "You answer questions about a supplied conversation. Use only the conversation. "
                        "Resolve relative times such as yesterday, last week, or next month into absolute dates or "
                        "date-anchored ranges using the relevant session header. For inference or yes/no questions, "
                        "give the best-supported inference rather than Unknown; for plural/list questions, include all "
                        "supported items. Return only the brief answer, with no explanation. If truly absent, say Unknown. /no_think"
                    ),
                },
                {
                    "role": "user",
                    "content": f"CONVERSATION\n{self.context}\n\nQUESTION\n{question}\n\nBRIEF ANSWER:",
                },
            ],
            seed_offset=question_index,
        )



class Mem0Adapter(MemoryAdapter):
    def __init__(self, client: OpenAIChatClient, condition: dict[str, Any], repo: Path):
        from mem0 import Memory

        self.client = client
        self.condition = condition
        self.conversation_index = 0
        self.user_id = ""
        self.ingestion_sessions = 0
        self.ingestion_events = 0
        store_path = repo / condition["store_path"]
        history_path = repo / condition["history_db_path"]
        if condition.get("reset_store", False):
            shutil.rmtree(store_path, ignore_errors=True)
            history_path.unlink(missing_ok=True)
        store_path.parent.mkdir(parents=True, exist_ok=True)
        self.memory = Memory.from_config(
            {
                "version": "v1.1",
                "history_db_path": str(history_path),
                "custom_instructions": condition.get("custom_instructions"),
                "llm": {
                    "provider": "openai",
                    "config": {
                        "model": condition["llm_model"],
                        "api_key": "local-no-cloud",
                        "openai_base_url": condition["llm_endpoint"],
                        "temperature": 0.0,
                        "max_tokens": int(condition["llm_max_tokens"]),
                    },
                },
                "embedder": {
                    "provider": "fastembed",
                    "config": {
                        "model": condition["embedding_model"],
                        "embedding_dims": int(condition["embedding_dims"]),
                    },
                },
                "vector_store": {
                    "provider": "qdrant",
                    "config": {
                        "collection_name": condition["collection_name"],
                        "embedding_model_dims": int(condition["embedding_dims"]),
                        "path": str(store_path),
                        "on_disk": True,
                    },
                },
            }
        )

    def ingest_conversation(self, conversation: dict[str, Any]) -> None:
        self.user_id = f'{self.condition["user_prefix"]}-{self.conversation_index}'
        self.conversation_index += 1
        self.ingestion_sessions = 0
        self.ingestion_events = 0
        speaker_roles: dict[str, str] = {}
        session_numbers = sorted(
            int(key.removeprefix("session_"))
            for key, value in conversation.items()
            if key.startswith("session_") and key.removeprefix("session_").isdigit() and isinstance(value, list)
        )
        for number in session_numbers:
            date = str(conversation.get(f"session_{number}_date_time", "date unavailable"))
            messages: list[dict[str, str]] = []
            for turn_index, turn in enumerate(conversation[f"session_{number}"]):
                speaker = str(turn["speaker"])
                if speaker not in speaker_roles:
                    speaker_roles[speaker] = "user" if not speaker_roles else "assistant"
                prefix = f"[Session {number} — {date}]\n" if turn_index == 0 else ""
                messages.append({"role": speaker_roles[speaker], "content": f'{prefix}{speaker}: {turn["text"]}'})
            result = self.memory.add(
                messages,
                user_id=self.user_id,
                metadata={"session": number, "date": date},
                infer=True,
            )
            self.ingestion_sessions += 1
            self.ingestion_events += len(result.get("results", []))

    def answer(self, question: str, *, question_index: int) -> dict[str, Any]:
        if not self.user_id:
            raise RuntimeError("ingest_conversation must be called before answer")
        search = self.memory.search(
            question,
            filters={"user_id": self.user_id},
            top_k=int(self.condition["top_k"]),
            threshold=float(self.condition["threshold"]),
        )
        memories = [
            {"memory": item["memory"], "score": item.get("score")}
            for item in search.get("results", [])
        ]
        context = "\n".join(f'- {item["memory"]}' for item in memories) or "(no memories retrieved)"
        response = self.client.chat(
            [
                {
                    "role": "system",
                    "content": (
                        "Answer the question using only the retrieved Mem0 memories. Resolve relative times when the "
                        "memory supplies a date. Give the best-supported brief answer, with no explanation. If absent, "
                        "say Unknown. /no_think"
                    ),
                },
                {"role": "user", "content": f"RETRIEVED MEMORIES\n{context}\n\nQUESTION\n{question}\n\nBRIEF ANSWER:"},
            ],
            seed_offset=question_index,
        )
        response["retrieval"] = {
            "user_id": self.user_id,
            "top_k": int(self.condition["top_k"]),
            "threshold": float(self.condition["threshold"]),
            "memories": memories,
            "ingestion_sessions": self.ingestion_sessions,
            "ingestion_events": self.ingestion_events,
        }
        return response


class _Mem0ParseErrorCounter(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.count = 0

    def emit(self, record: logging.LogRecord) -> None:
        if record.getMessage().startswith("Error parsing extraction response"):
            self.count += 1


class Mem0TurnAdapter(Mem0Adapter):
    """OSS condition matching the public runner's one-turn ingestion chunks."""

    def __init__(self, client: OpenAIChatClient, condition: dict[str, Any], repo: Path):
        super().__init__(client, condition, repo)
        self.repo = repo
        self.ingestion_turns = 0
        self.ingestion_parse_errors = 0
        self.ingestion_retries = 0
        self.ingestion_residual_failures = 0
        self.ingestion_empty_results = 0
        self.parse_counter = _Mem0ParseErrorCounter()
        logging.getLogger("mem0.memory.main").addHandler(self.parse_counter)
        self.ingestion_log_path = repo / condition["ingestion_log_path"]
        self.ingestion_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.ingestion_log_path.write_text("")
        self.json_repairs = 0
        self._repair_malformed_json = False
        original_generate = self.memory.llm.generate_response

        def generate_with_bounded_repair(*args: Any, **kwargs: Any) -> Any:
            response = original_generate(*args, **kwargs)
            if self._repair_malformed_json and kwargs.get("response_format") and isinstance(response, str):
                repaired = self._trim_extra_closing_braces(response)
                if repaired != response:
                    self.json_repairs += 1
                    return repaired
            return response

        self.memory.llm.generate_response = generate_with_bounded_repair

    @staticmethod
    def _trim_extra_closing_braces(value: str) -> str:
        from json_repair import repair_json
        from mem0.memory.utils import remove_code_blocks

        candidate = remove_code_blocks(value).strip()
        try:
            json.loads(candidate, strict=False)
            return candidate
        except json.JSONDecodeError:
            repaired = repair_json(candidate, ensure_ascii=False)
        try:
            json.loads(repaired, strict=False)
            return repaired
        except json.JSONDecodeError:
            return value

    _parse_date = staticmethod(parse_locomo_datetime)
    _turn_text = staticmethod(locomo_turn_text)

    def ingest_conversation(self, conversation: dict[str, Any]) -> None:
        conversation_index = self.conversation_index
        self.user_id = f'{self.condition["user_prefix"]}-{conversation_index}'
        self.conversation_index += 1
        self.ingestion_sessions = 0
        self.ingestion_events = 0
        self.ingestion_turns = 0
        self.ingestion_parse_errors = 0
        self.ingestion_retries = 0
        self.ingestion_residual_failures = 0
        self.ingestion_empty_results = 0
        speaker_a = str(conversation["speaker_a"])
        sessions: list[tuple[int, str, list[dict[str, Any]]]] = []
        for key, turns in conversation.items():
            suffix = key.removeprefix("session_")
            if key.startswith("session_") and suffix.isdigit() and isinstance(turns, list):
                number = int(suffix)
                date = str(conversation.get(f"session_{number}_date_time", ""))
                sessions.append((number, date, turns))
        sessions.sort(key=lambda item: (self._parse_date(item[1]) is None, self._parse_date(item[1]) or item[0]))

        for number, date, turns in sessions:
            parsed_date = self._parse_date(date)
            session_epoch = int(parsed_date.timestamp()) if parsed_date else None
            self.ingestion_sessions += 1
            for turn_index, turn in enumerate(turns):
                text = self._turn_text(turn)
                if not text:
                    continue
                speaker = str(turn.get("speaker", ""))
                role = "user" if speaker == speaker_a else "assistant"
                message = {
                    "role": role,
                    "content": f"[Authoritative message timestamp: {date}]\n{speaker}: {text}",
                }
                metadata = {
                    "session": number,
                    "date": date,
                    "session_epoch": session_epoch,
                    "source_type": "prime",
                    "timestamp_transport": "metadata-and-content-prefix",
                }
                attempts = 0
                parse_errors = 0
                error: str | None = None
                result: dict[str, Any] = {"results": []}
                attempt_details: list[dict[str, Any]] = []
                base_max_tokens = int(self.condition["llm_max_tokens"])
                while attempts < 2:
                    attempts += 1
                    max_tokens = (
                        base_max_tokens
                        if attempts == 1
                        else int(self.condition["retry_llm_max_tokens"])
                    )
                    self.memory.llm.config.max_tokens = max_tokens
                    self._repair_malformed_json = attempts == 2
                    repair_before = self.json_repairs
                    before = self.parse_counter.count
                    error = None
                    try:
                        result = self.memory.add(
                            [message],
                            user_id=self.user_id,
                            metadata=metadata,
                            infer=True,
                        )
                    except Exception as exc:
                        error = f"{type(exc).__name__}: {exc}"
                    attempt_parse_errors = self.parse_counter.count - before
                    parse_errors += attempt_parse_errors
                    attempt_details.append(
                        {
                            "attempt": attempts,
                            "max_tokens": max_tokens,
                            "parse_errors": attempt_parse_errors,
                            "json_repairs": self.json_repairs - repair_before,
                            "error": error,
                        }
                    )
                    if error is None and attempt_parse_errors == 0:
                        break
                    if attempts < 2:
                        self.ingestion_retries += 1
                self.memory.llm.config.max_tokens = base_max_tokens
                self._repair_malformed_json = False
                residual_failure = error is not None or attempt_parse_errors > 0
                result_items = result.get("results", []) if not residual_failure else []
                events = len(result_items)
                self.ingestion_turns += 1
                self.ingestion_events += events
                self.ingestion_parse_errors += parse_errors
                self.ingestion_residual_failures += int(residual_failure)
                self.ingestion_empty_results += int(not residual_failure and events == 0)
                input_payload = {"message": message, "metadata": metadata}
                input_sha256 = hashlib.sha256(
                    json.dumps(input_payload, sort_keys=True, ensure_ascii=False).encode()
                ).hexdigest()
                record = {
                    "conversation_index": conversation_index,
                    "user_id": self.user_id,
                    "session": number,
                    "session_date": date,
                    "session_epoch": session_epoch,
                    "turn_index": turn_index,
                    "speaker": speaker,
                    "role": role,
                    "message": message,
                    "metadata": metadata,
                    "input_sha256": input_sha256,
                    "attempts": attempts,
                    "attempt_details": attempt_details,
                    "parse_errors": parse_errors,
                    "json_repairs": sum(item["json_repairs"] for item in attempt_details),
                    "residual_failure": residual_failure,
                    "error": error,
                    "events": events,
                    "event_content_hashes": [
                        hashlib.sha256(
                            json.dumps(item, sort_keys=True, ensure_ascii=False, default=str).encode()
                        ).hexdigest()
                        for item in result_items
                    ],
                }
                with self.ingestion_log_path.open("a") as log:
                    log.write(json.dumps(record, ensure_ascii=False) + "\n")
                if residual_failure and self.condition.get("fail_on_residual", True):
                    raise RuntimeError(
                        f"Mem0 ingestion remained malformed after retry: conversation={conversation_index} "
                        f"session={number} turn={turn_index}; see {self.ingestion_log_path}"
                    )


    def answer(self, question: str, *, question_index: int) -> dict[str, Any]:
        response = super().answer(question, question_index=question_index)
        response["retrieval"].update(
            {
                "ingestion_turns": self.ingestion_turns,
                "ingestion_parse_errors": self.ingestion_parse_errors,
                "ingestion_retries": self.ingestion_retries,
                "ingestion_residual_failures": self.ingestion_residual_failures,
                "ingestion_empty_results": self.ingestion_empty_results,
                "timestamp_transport": "metadata-and-content-prefix (OSS rejects timestamp parameter)",
            }
        )
        return response

    def receipt_metadata(self) -> dict[str, Any]:
        rows = [json.loads(line) for line in self.ingestion_log_path.read_text().splitlines() if line]
        by_conversation: dict[str, dict[str, int]] = {}
        for row in rows:
            key = str(row["conversation_index"])
            summary = by_conversation.setdefault(
                key,
                {
                    "turns": 0,
                    "events": 0,
                    "parse_errors": 0,
                    "json_repairs": 0,
                    "retries": 0,
                    "residual_failures": 0,
                    "empty_results": 0,
                },
            )
            summary["turns"] += 1
            summary["events"] += int(row["events"])
            summary["parse_errors"] += int(row["parse_errors"])
            summary["json_repairs"] += int(row.get("json_repairs", 0))
            summary["retries"] += int(row["attempts"] > 1)
            summary["residual_failures"] += int(row["residual_failure"])
            summary["empty_results"] += int(not row["residual_failure"] and row["events"] == 0)
        data = self.ingestion_log_path.read_bytes()
        return {
            "ingestion": {
                "log_path": str(self.ingestion_log_path.relative_to(self.repo)),
                "log_sha256": hashlib.sha256(data).hexdigest(),
                "turns": len(rows),
                "events": sum(int(row["events"]) for row in rows),
                "parse_errors": sum(int(row["parse_errors"]) for row in rows),
                "json_repairs": sum(int(row.get("json_repairs", 0)) for row in rows),
                "retries": sum(int(row["attempts"] > 1) for row in rows),
                "residual_failures": sum(int(row["residual_failure"]) for row in rows),
                "empty_results": sum(
                    int(not row["residual_failure"] and row["events"] == 0) for row in rows
                ),
                "by_conversation": by_conversation,
                "complete": bool(rows) and not any(row["residual_failure"] for row in rows),
            }
        }


class MemMachineAdapter(MemoryAdapter):
    """MemMachine OSS condition, wired entirely to local services.

    The in-process wiring mirrors MemMachine's own LoCoMo benchmark
    (`evaluation/retrieval_agent/locomo_ingest.py`, `locomo_search.py` and
    `evaluation/utils/agent_utils.py::init_memmachine_params` at the pinned
    commit): a `ResourceManagerImpl` built from a MemMachine YAML configuration,
    a declarative `LongTermMemory` over a local Neo4j, and the `MemMachineAgent`
    retriever, which performs a plain `query_memory` with no extra LLM calls.
    Only the answer/judge protocol is the harness's, so every condition is
    graded identically.
    """

    def __init__(self, client: OpenAIChatClient, condition: dict[str, Any], repo: Path):
        import yaml
        from memmachine_server.common.configuration import Configuration
        from memmachine_server.common.metrics_factory import PrometheusMetricsFactory
        from memmachine_server.common.resource_manager.resource_manager import ResourceManagerImpl
        from memmachine_server.episodic_memory import EpisodicMemory
        from memmachine_server.episodic_memory.episodic_memory import EpisodicMemoryParams
        from memmachine_server.episodic_memory.long_term_memory import DeclarativeBackendParams, LongTermMemory
        from memmachine_server.retrieval_agent.agents import MemMachineAgent
        from memmachine_server.retrieval_agent.common.agent_api import AgentToolBaseParam

        from .local_embeddings import LocalEmbeddingService

        self.client = client
        self.condition = condition
        self.repo = repo
        self.conversation_index = 0
        self.session_key = ""
        self.ingestion_turns = 0
        self.ingestion_sessions = 0
        self.ingestion_failures = 0
        self.max_sessions = condition.get("max_sessions")

        password = os.environ.get(str(condition["graph_password_env"]), "")
        if not password:
            raise RuntimeError(
                f'environment variable {condition["graph_password_env"]} must hold the local Neo4j password'
            )

        self.embeddings = LocalEmbeddingService(
            str(condition["embedding_model"]), int(condition["embedding_dims"])
        )
        embedding_base_url = self.embeddings.start_http()

        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        if condition.get("reset_store", False):
            self.loop.run_until_complete(self._reset_graph(password))

        runtime_directory = (repo / str(condition["sqlite_path"])).parent
        runtime_directory.mkdir(parents=True, exist_ok=True)
        rendered = self._render_memmachine_config(
            condition, embedding_base_url, password, runtime_directory / "memmachine.log"
        )
        # The live configuration carries the local Neo4j password, so it stays in the
        # ignored .bench working directory; only a redacted copy is receipted.
        self.config_path = runtime_directory / "memmachine-config.yaml"
        self.config_path.write_text(yaml.safe_dump(rendered, sort_keys=True))
        redacted = copy.deepcopy(rendered)
        redacted["resources"]["databases"]["locomo_graph"]["config"]["password"] = "<redacted>"
        redacted_yaml = yaml.safe_dump(redacted, sort_keys=True)
        self.config_sha256 = hashlib.sha256(redacted_yaml.encode()).hexdigest()
        evidence_path = repo / condition["memmachine_config_path"]
        evidence_path.parent.mkdir(parents=True, exist_ok=True)
        evidence_path.write_text(redacted_yaml)

        configuration = Configuration.load_yml_file(str(self.config_path))
        self.resource_manager = ResourceManagerImpl(configuration)

        async def build() -> tuple[Any, Any, Any]:
            embedder = await self.resource_manager.get_embedder("locomo_embedder")
            reranker = await self.resource_manager.get_reranker("locomo_reranker")
            vector_graph_store = await self.resource_manager.get_vector_graph_store("locomo_graph")
            return embedder, reranker, vector_graph_store

        self._embedder, self._reranker, self._vector_graph_store = self.loop.run_until_complete(build())
        self._memory_factory = lambda session_key: EpisodicMemory(
            EpisodicMemoryParams(
                session_key=session_key,
                metrics_factory=PrometheusMetricsFactory(),
                long_term_memory=LongTermMemory(
                    DeclarativeBackendParams(
                        session_id=session_key,
                        vector_graph_store=self._vector_graph_store,
                        embedder=self._embedder,
                        reranker=self._reranker,
                        message_sentence_chunking=bool(condition.get("message_sentence_chunking", False)),
                    )
                ),
                short_term_memory=None,
                enabled=True,
            )
        )
        self._agent = MemMachineAgent(
            AgentToolBaseParam(model=None, children_tools=[], extra_params={}, reranker=self._reranker)
        )
        self.memory: Any = None

        self.ingestion_log_path = repo / condition["ingestion_log_path"]
        self.ingestion_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.ingestion_log_path.write_text("")

    async def _reset_graph(self, password: str) -> None:
        from neo4j import AsyncGraphDatabase

        driver = AsyncGraphDatabase.driver(
            str(self.condition["graph_uri"]), auth=(str(self.condition["graph_user"]), password)
        )
        try:
            async with driver.session() as session:
                await session.run("MATCH (n) DETACH DELETE n")
        finally:
            await driver.close()

    @staticmethod
    def _render_memmachine_config(
        condition: dict[str, Any], embedding_base_url: str, password: str, log_path: Path
    ) -> dict[str, Any]:
        """Build the MemMachine YAML configuration for a fully local run."""
        return {
            "logging": {"level": "warning", "path": str(log_path)},
            "episode_store": {"database": "locomo_sqlite"},
            "session_manager": {"database": "locomo_sqlite"},
            "semantic_memory": {"enabled": False, "config_database": "locomo_sqlite"},
            "episodic_memory": {
                "enabled": True,
                "long_term_memory_enabled": True,
                "short_term_memory_enabled": False,
                "long_term_memory": {
                    "embedder": "locomo_embedder",
                    "reranker": "locomo_reranker",
                    "vector_graph_store": "locomo_graph",
                },
            },
            "retrieval_agent": {"llm_model": "locomo_llm", "reranker": "locomo_reranker"},
            "resources": {
                "databases": {
                    "locomo_graph": {
                        "provider": "neo4j",
                        "config": {
                            "uri": str(condition["graph_uri"]),
                            "user": str(condition["graph_user"]),
                            "password": password,
                        },
                    },
                    "locomo_sqlite": {
                        "provider": "sqlite",
                        "config": {
                            "dialect": "sqlite",
                            "driver": "aiosqlite",
                            "path": str(condition["sqlite_path"]),
                        },
                    },
                },
                "embedders": {
                    "locomo_embedder": {
                        "provider": "openai",
                        "config": {
                            "model": str(condition["embedding_model"]),
                            "api_key": "local-no-cloud",
                            "base_url": embedding_base_url,
                            "dimensions": int(condition["embedding_dims"]),
                            "max_input_length": int(condition["embedding_max_input_length"]),
                        },
                    }
                },
                "language_models": {
                    "locomo_llm": {
                        "provider": "openai-chat-completions",
                        "config": {
                            "model": str(condition["llm_model"]),
                            "api_key": "local-no-cloud",
                            "base_url": str(condition["llm_endpoint"]),
                        },
                    }
                },
                "rerankers": {
                    "locomo_reranker": {
                        "provider": "bm25",
                        "config": {
                            "k1": float(condition["bm25_k1"]),
                            "b": float(condition["bm25_b"]),
                            "epsilon": float(condition["bm25_epsilon"]),
                            "language": str(condition["bm25_language"]),
                            "tokenizer": str(condition["bm25_tokenizer"]),
                        },
                    }
                },
            },
        }

    def ingest_conversation(self, conversation: dict[str, Any]) -> None:
        from memmachine_server.common.episode_store import Episode

        conversation_index = self.conversation_index
        self.conversation_index += 1
        self.session_key = f'{self.condition["session_prefix"]}-{conversation_index}'
        self.memory = self._memory_factory(self.session_key)
        self.ingestion_turns = 0
        self.ingestion_sessions = 0
        self.ingestion_failures = 0

        for number, date, turns in chronological_sessions(conversation, self.max_sessions):
            session_datetime = parse_locomo_datetime(date)
            self.ingestion_sessions += 1
            for turn_index, turn in enumerate(turns):
                text = locomo_turn_text(turn)
                if not text:
                    continue
                speaker = str(turn.get("speaker", ""))
                created_at = (
                    session_datetime + timedelta(seconds=turn_index)
                    if session_datetime
                    else datetime.now(timezone.utc)
                )
                episode = Episode(
                    uid=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{self.session_key}/{number}/{turn_index}")),
                    content=text,
                    session_key=self.session_key,
                    created_at=created_at,
                    producer_id=speaker,
                    producer_role=speaker,
                    metadata={
                        "locomo_session_id": f"session_{number}",
                        "locomo_session_date": date,
                        "locomo_turn_index": turn_index,
                        "locomo_dialog_id": str(turn.get("dia_id", "")),
                    },
                )
                error: str | None = None
                started = time.time()
                try:
                    self.loop.run_until_complete(self.memory.add_memory_episodes([episode]))
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    self.ingestion_failures += 1
                self.ingestion_turns += 1
                with self.ingestion_log_path.open("a") as log:
                    log.write(
                        json.dumps(
                            {
                                "conversation_index": conversation_index,
                                "session_key": self.session_key,
                                "session": number,
                                "session_date": date,
                                "turn_index": turn_index,
                                "speaker": speaker,
                                "episode_uid": episode.uid,
                                "created_at": created_at.isoformat(),
                                "characters": len(text),
                                "seconds": round(time.time() - started, 4),
                                "error": error,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

    def answer(self, question: str, *, question_index: int) -> dict[str, Any]:
        from memmachine_server.common.episode_store.episode_model import episodes_to_string
        from memmachine_server.retrieval_agent.common.agent_api import QueryParam, QueryPolicy

        if self.memory is None:
            raise RuntimeError("ingest_conversation must be called before answer")
        started = time.time()
        episodes, perf_metrics = self.loop.run_until_complete(
            self._agent.do_query(
                QueryPolicy(
                    token_cost=10,
                    time_cost=10,
                    accuracy_score=10,
                    confidence_score=10,
                    max_attempts=3,
                    max_return_len=int(self.condition["max_return_len"]),
                ),
                QueryParam(
                    query=question,
                    limit=int(self.condition["top_k"]),
                    memory=self.memory,
                    expand_context=int(self.condition["expand_context"]),
                ),
            )
        )
        retrieval_seconds = time.time() - started
        context = episodes_to_string(episodes) or "(no episodes retrieved)"
        response = self.client.chat(
            [
                {
                    "role": "system",
                    "content": (
                        "Answer the question using only the retrieved MemMachine episodes. Resolve relative times when "
                        "the episode supplies a date. Give the best-supported brief answer, with no explanation. If "
                        "absent, say Unknown. /no_think"
                    ),
                },
                {"role": "user", "content": f"RETRIEVED EPISODES\n{context}\n\nQUESTION\n{question}\n\nBRIEF ANSWER:"},
            ],
            seed_offset=question_index,
        )
        response["retrieval"] = {
            "session_key": self.session_key,
            "top_k": int(self.condition["top_k"]),
            "expand_context": int(self.condition["expand_context"]),
            "retrieval_seconds": round(retrieval_seconds, 4),
            "agent": perf_metrics.get("agent"),
            "memory_search_called": perf_metrics.get("memory_search_called"),
            "episodes": [
                {
                    "uid": str(episode.uid),
                    "producer_id": episode.producer_id,
                    "created_at": episode.created_at.isoformat() if episode.created_at else None,
                    "content": episode.content,
                    "metadata": episode.metadata,
                }
                for episode in episodes
            ],
            "ingestion_sessions": self.ingestion_sessions,
            "ingestion_turns": self.ingestion_turns,
            "ingestion_failures": self.ingestion_failures,
            "memmachine_config_sha256": self.config_sha256,
            "embedder": self.embeddings.disclosure(),
        }
        return response


class GraphitiAdapter(MemoryAdapter):
    """Graphiti OSS condition on an embedded FalkorDB Lite graph, fully local.

    Wiring follows the pinned repository's own recipes
    (`examples/podcast/podcast_runner.py` for the embedded FalkorDB Lite client,
    `graphiti_core.search.search_config_recipes` for hybrid search). Extraction
    runs through `OpenAIGenericClient` against the local llama.cpp endpoint,
    embeddings are computed in-process on CPU, and the default cloud reranker is
    replaced by an order-preserving local stub that the RRF/MMR recipes never
    call. Telemetry is disabled before `graphiti_core` is imported.
    """

    def __init__(self, client: OpenAIChatClient, condition: dict[str, Any], repo: Path):
        self.client = client
        self.condition = condition
        self.repo = repo
        self.conversation_index = 0
        self.group_id = ""
        self.ingestion_sessions = 0
        self.ingestion_turns = 0
        self.ingestion_failures = 0
        self.ingestion_nodes = 0
        self.ingestion_edges = 0
        self.max_sessions = condition.get("max_sessions")
        self.max_turns_per_session = condition.get("max_turns_per_session")

        os.environ["GRAPHITI_TELEMETRY_ENABLED"] = "false"
        os.environ["EMBEDDING_DIM"] = str(int(condition["embedding_dims"]))
        os.environ["SEMAPHORE_LIMIT"] = str(int(condition["semaphore_limit"]))
        os.environ.setdefault("OPENAI_API_KEY", "local-no-cloud")

        from graphiti_core import Graphiti
        from graphiti_core.cross_encoder.client import CrossEncoderClient
        from graphiti_core.driver.falkordb_driver import FalkorDriver
        from graphiti_core.embedder.client import EmbedderClient
        from graphiti_core.llm_client.config import LLMConfig
        from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
        from redislite.async_falkordb_client import AsyncFalkorDB

        from .local_embeddings import LocalEmbeddingService

        self.embeddings = LocalEmbeddingService(
            str(condition["embedding_model"]), int(condition["embedding_dims"])
        )
        embeddings = self.embeddings

        class LocalFastEmbedEmbedder(EmbedderClient):
            async def create(self, input_data: Any) -> list[float]:
                texts = [input_data] if isinstance(input_data, str) else [str(item) for item in input_data]
                return embeddings.embed(texts)[0]

            async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
                return embeddings.embed([str(item) for item in input_data_list])

        class OrderPreservingReranker(CrossEncoderClient):
            """Local, deterministic stub. RRF and MMR recipes never invoke rank()."""

            calls = 0

            async def rank(self, query: str, passages: list[str]) -> list[tuple[str, float]]:
                OrderPreservingReranker.calls += 1
                total = max(len(passages), 1)
                return [(passage, 1.0 - index / total) for index, passage in enumerate(passages)]

        self._reranker_class = OrderPreservingReranker
        store_path = repo / condition["store_path"]
        if condition.get("reset_store", False):
            shutil.rmtree(store_path.parent, ignore_errors=True)
        store_path.parent.mkdir(parents=True, exist_ok=True)

        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.falkor_db = AsyncFalkorDB(dbfilename=str(store_path))
        self.driver = FalkorDriver(falkor_db=self.falkor_db)
        self.llm_config = LLMConfig(
            api_key="local-no-cloud",
            model=str(condition["llm_model"]),
            small_model=str(condition["llm_model"]),
            base_url=str(condition["llm_endpoint"]),
            temperature=float(condition["llm_temperature"]),
            max_tokens=int(condition["llm_max_tokens"]),
        )
        self.graphiti = Graphiti(
            graph_driver=self.driver,
            llm_client=OpenAIGenericClient(
                config=self.llm_config,
                max_tokens=int(condition["llm_max_tokens"]),
                structured_output_mode=str(condition["structured_output_mode"]),
            ),
            embedder=LocalFastEmbedEmbedder(),
            cross_encoder=OrderPreservingReranker(),
            max_coroutines=int(condition["max_coroutines"]),
        )
        self.loop.run_until_complete(self.graphiti.build_indices_and_constraints())

        self.ingestion_log_path = repo / condition["ingestion_log_path"]
        self.ingestion_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.ingestion_log_path.write_text("")

    def _search_config(self) -> Any:
        from graphiti_core.search import search_config_recipes

        recipe = copy.deepcopy(getattr(search_config_recipes, str(self.condition["search_recipe"])))
        recipe.limit = int(self.condition["search_limit"])
        if recipe.edge_config is not None:
            recipe.edge_config.sim_min_score = recipe.edge_config.sim_min_score
        return recipe

    def ingest_conversation(self, conversation: dict[str, Any]) -> None:
        from graphiti_core.nodes import EpisodeType

        conversation_index = self.conversation_index
        self.conversation_index += 1
        self.group_id = f'{self.condition["group_prefix"]}_{conversation_index}'
        self.ingestion_sessions = 0
        self.ingestion_turns = 0
        self.ingestion_failures = 0
        self.ingestion_nodes = 0
        self.ingestion_edges = 0

        for number, date, turns in chronological_sessions(conversation, self.max_sessions):
            session_datetime = parse_locomo_datetime(date)
            self.ingestion_sessions += 1
            selected = turns if self.max_turns_per_session is None else turns[: int(self.max_turns_per_session)]
            for turn_index, turn in enumerate(selected):
                text = locomo_turn_text(turn)
                if not text:
                    continue
                speaker = str(turn.get("speaker", ""))
                reference_time = (
                    session_datetime + timedelta(seconds=turn_index)
                    if session_datetime
                    else datetime(1970, 1, 1, tzinfo=timezone.utc)
                    + timedelta(days=number, seconds=turn_index)
                )
                started = time.time()
                error: str | None = None
                nodes = edges = 0
                try:
                    result = self.loop.run_until_complete(
                        self.graphiti.add_episode(
                            name=f"{self.group_id}-s{number}-t{turn_index}",
                            episode_body=f"{speaker}: {text}",
                            source_description=(
                                f"LoCoMo session {number} recorded {date}" if date else f"LoCoMo session {number}"
                            ),
                            reference_time=reference_time,
                            source=EpisodeType.message,
                            group_id=self.group_id,
                        )
                    )
                    nodes = len(result.nodes)
                    edges = len(result.edges)
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    self.ingestion_failures += 1
                self.ingestion_turns += 1
                self.ingestion_nodes += nodes
                self.ingestion_edges += edges
                with self.ingestion_log_path.open("a") as log:
                    log.write(
                        json.dumps(
                            {
                                "conversation_index": conversation_index,
                                "group_id": self.group_id,
                                "session": number,
                                "session_date": date,
                                "turn_index": turn_index,
                                "speaker": speaker,
                                "reference_time": reference_time.isoformat(),
                                "characters": len(text),
                                "nodes": nodes,
                                "edges": edges,
                                "seconds": round(time.time() - started, 4),
                                "error": error,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
        if self.ingestion_failures:
            raise RuntimeError(
                f"Graphiti ingestion failed for {self.ingestion_failures}/{self.ingestion_turns} turns; see {self.ingestion_log_path}"
            )

    def answer(self, question: str, *, question_index: int) -> dict[str, Any]:
        from graphiti_core.search.search_helpers import search_results_to_context_string

        if not self.group_id:
            raise RuntimeError("ingest_conversation must be called before answer")
        search_config = self._search_config()
        started = time.time()
        results = self.loop.run_until_complete(
            self.graphiti.search_(question, config=search_config, group_ids=[self.group_id])
        )
        retrieval_seconds = time.time() - started
        context = search_results_to_context_string(results) or "(no graph facts retrieved)"
        response = self.client.chat(
            [
                {
                    "role": "system",
                    "content": (
                        "Answer the question using only the retrieved Graphiti graph facts and entities. Resolve "
                        "relative times when a fact supplies a date. Give the best-supported brief answer, with no "
                        "explanation. If absent, say Unknown. /no_think"
                    ),
                },
                {"role": "user", "content": f"RETRIEVED GRAPH CONTEXT\n{context}\n\nQUESTION\n{question}\n\nBRIEF ANSWER:"},
            ],
            seed_offset=question_index,
        )
        response["retrieval"] = {
            "group_id": self.group_id,
            "search_recipe": str(self.condition["search_recipe"]),
            "search_limit": search_config.limit,
            "limits": {
                "edge_config": search_config.edge_config.model_dump(mode="json") if search_config.edge_config else None,
                "node_config": search_config.node_config.model_dump(mode="json") if search_config.node_config else None,
                "episode_config": search_config.episode_config.model_dump(mode="json") if search_config.episode_config else None,
                "community_config": search_config.community_config.model_dump(mode="json")
                if search_config.community_config
                else None,
                "reranker_min_score": search_config.reranker_min_score,
            },
            "retrieval_seconds": round(retrieval_seconds, 4),
            "counts": {
                "edges": len(results.edges),
                "nodes": len(results.nodes),
                "episodes": len(results.episodes),
                "communities": len(results.communities),
            },
            "edges": [
                {
                    "uuid": edge.uuid,
                    "name": edge.name,
                    "fact": edge.fact,
                    "valid_at": edge.valid_at.isoformat() if edge.valid_at else None,
                    "invalid_at": edge.invalid_at.isoformat() if edge.invalid_at else None,
                }
                for edge in results.edges
            ],
            "nodes": [{"uuid": node.uuid, "name": node.name, "summary": node.summary} for node in results.nodes],
            "reranker_calls": self._reranker_class.calls,
            "ingestion_sessions": self.ingestion_sessions,
            "ingestion_turns": self.ingestion_turns,
            "ingestion_failures": self.ingestion_failures,
            "ingestion_nodes": self.ingestion_nodes,
            "ingestion_edges": self.ingestion_edges,
            "embedder": self.embeddings.disclosure(),
            "telemetry": os.environ.get("GRAPHITI_TELEMETRY_ENABLED"),
        }
        return response

    def receipt_metadata(self) -> dict[str, Any]:
        data = self.ingestion_log_path.read_bytes() if self.ingestion_log_path.exists() else b""
        return {
            "graphiti_repository": self.condition["graphiti_repository"],
            "graphiti_commit": self.condition["graphiti_commit"],
            "graph_store": self.condition["graph_store"],
            "telemetry": os.environ.get("GRAPHITI_TELEMETRY_ENABLED"),
            "ingestion_log_path": str(self.ingestion_log_path.relative_to(self.repo)),
            "ingestion_log_sha256": hashlib.sha256(data).hexdigest(),
            "ingestion_turns": self.ingestion_turns,
            "ingestion_failures": self.ingestion_failures,
            "ingestion_nodes": self.ingestion_nodes,
            "ingestion_edges": self.ingestion_edges,
            "embedder": self.embeddings.disclosure(),
        }






class HindsightAdapter(MemoryAdapter):
    """Hindsight OSS through its self-hosted loopback REST API.

    The server is a separate pinned process because its embedded PostgreSQL and
    model stack have a different lifecycle from the benchmark runner. The
    adapter only accepts its configured endpoint; queued runs bind it to
    loopback. The queued configuration uses Hindsight 0.9.1 with embedded pg0,
    local CPU embeddings/reranking, and the resident llama.cpp endpoint.
    """

    def __init__(self, client: OpenAIChatClient, condition: dict[str, Any], repo: Path):
        self.client = client
        self.condition = condition
        self.repo = repo
        self.base_url = str(condition["api_endpoint"]).rstrip("/")
        self.conversation_index = 0
        self.bank_id = ""
        self.ingestion_sessions = 0
        self.ingestion_turns = 0
        self.ingestion_failures = 0
        self.total_ingestion_sessions = 0
        self.total_ingestion_turns = 0
        self.total_ingestion_failures = 0
        self.bank_ids: list[str] = []
        self.query_timestamp: str | None = None
        self.max_sessions = condition.get("max_sessions")
        self.max_turns_per_session = condition.get("max_turns_per_session")
        self.ingestion_log_path = repo / condition["ingestion_log_path"]
        self.ingestion_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.ingestion_log_path.write_text("")
        health = self._request("GET", "/health")
        version = self._request("GET", "/version")
        if not isinstance(health, dict) or not isinstance(version, dict):
            raise RuntimeError("Hindsight health/version endpoint returned a non-object response")
        if health.get("status") != "healthy" or health.get("database") != "connected":
            raise RuntimeError(f"Hindsight server is not healthy and connected: {health}")
        observed_version = str(version.get("api_version", version.get("version", "")))
        if observed_version != str(condition["expected_package_version"]):
            raise RuntimeError(
                f"Hindsight server version mismatch: expected {condition['expected_package_version']}, got {observed_version}"
            )
        if condition.get("require_bank_llm_health", False) and not version.get("features", {}).get(
            "bank_llm_health", False
        ):
            raise RuntimeError("Hindsight server does not expose required bank LLM health evidence")
        self.health = health
        self.version = version

    def _request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> Any:
        import urllib.error
        import urllib.request

        data = json.dumps(payload).encode() if payload is not None else None
        request = urllib.request.Request(
            self.base_url + path,
            data=data,
            method=method,
            headers={"Content-Type": "application/json", "User-Agent": "ax0s-memory-bench/0.0.1"},
        )
        try:
            with urllib.request.urlopen(request, timeout=int(self.condition["timeout_seconds"])) as response:
                body = response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")
            raise RuntimeError(f"Hindsight {method} {path} failed with HTTP {exc.code}: {detail}") from exc
        return json.loads(body) if body else {}

    def ingest_conversation(self, conversation: dict[str, Any]) -> None:
        conversation_index = self.conversation_index
        self.conversation_index += 1
        self.bank_id = f'{self.condition["bank_prefix"]}-{conversation_index}'
        self.bank_ids.append(self.bank_id)
        self.ingestion_sessions = 0
        self.ingestion_turns = 0
        self.ingestion_failures = 0
        if self.condition.get("reset_bank", False):
            cleared = self._request("DELETE", f"/v1/default/banks/{self.bank_id}/memories")
            if cleared.get("success") is not True:
                raise RuntimeError(f"Hindsight bank reset was not confirmed: {cleared}")
        items: list[dict[str, Any]] = []
        log_rows: list[dict[str, Any]] = []
        conversation_last_timestamp: datetime | None = None

        for number, date, turns in chronological_sessions(conversation, self.max_sessions):
            self.ingestion_sessions += 1
            session_datetime = parse_locomo_datetime(date)
            selected = turns if self.max_turns_per_session is None else turns[: int(self.max_turns_per_session)]
            for turn_index, turn in enumerate(selected):
                text = locomo_turn_text(turn)
                if not text:
                    continue
                speaker = str(turn.get("speaker", ""))
                timestamp = (
                    session_datetime + timedelta(seconds=turn_index)
                    if session_datetime
                    else datetime(1970, 1, 1, tzinfo=timezone.utc)
                    + timedelta(days=number, seconds=turn_index)
                )
                conversation_last_timestamp = max(conversation_last_timestamp or timestamp, timestamp)
                items.append(
                    {
                        "content": f"[Session {number} — {date}]\n{speaker}: {text}",
                        "timestamp": timestamp.isoformat(),
                        "context": f"LoCoMo conversation {conversation_index}, session {number}",
                        "metadata": {
                            "conversation_index": str(conversation_index),
                            "session": str(number),
                            "turn_index": str(turn_index),
                            "speaker": speaker,
                        },
                    }
                )
                log_rows.append(
                    {
                        "conversation_index": conversation_index,
                        "bank_id": self.bank_id,
                        "session": number,
                        "session_date": date,
                        "turn_index": turn_index,
                        "speaker": speaker,
                        "timestamp": timestamp.isoformat(),
                        "characters": len(text),
                    }
                )
        if not items or conversation_last_timestamp is None:
            raise RuntimeError("Hindsight ingestion received no nonempty turns")
        self.query_timestamp = conversation_last_timestamp.isoformat()
        self.total_ingestion_sessions += self.ingestion_sessions
        batch_size = int(self.condition["retain_batch_size"])
        if batch_size < 1:
            raise RuntimeError("Hindsight retain_batch_size must be positive")
        self.ingestion_turns = 0
        with self.ingestion_log_path.open("a") as log:
            for start in range(0, len(items), batch_size):
                batch = items[start : start + batch_size]
                batch_rows = log_rows[start : start + batch_size]
                started = time.time()
                try:
                    result = self._request(
                        "POST",
                        f"/v1/default/banks/{self.bank_id}/memories",
                        {"items": batch, "async": False},
                    )
                    if not result.get("success") or int(result.get("items_count", -1)) != len(batch):
                        raise RuntimeError(f"Hindsight retain did not confirm all items: {result}")
                except Exception as exc:
                    seconds = round(time.time() - started, 4)
                    self.ingestion_failures += len(batch)
                    self.total_ingestion_failures += len(batch)
                    for row in batch_rows:
                        row.update(
                            {
                                "retain_batch_items": len(batch),
                                "retain_batch_seconds": seconds,
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                        )
                        log.write(json.dumps(row, ensure_ascii=False) + "\n")
                    log.flush()
                    raise
                seconds = round(time.time() - started, 4)
                self.ingestion_turns += len(batch)
                self.total_ingestion_turns += len(batch)
                for row in batch_rows:
                    row.update({"retain_batch_items": len(batch), "retain_batch_seconds": seconds, "error": None})
                    log.write(json.dumps(row, ensure_ascii=False) + "\n")
                log.flush()

    def answer(self, question: str, *, question_index: int) -> dict[str, Any]:
        if not self.bank_id:
            raise RuntimeError("ingest_conversation must be called before answer")
        started = time.time()
        recall = self._request(
            "POST",
            f"/v1/default/banks/{self.bank_id}/memories/recall",
            {
                "query": question,
                "types": ["world", "experience"],
                "budget": str(self.condition["recall_budget"]),
                "max_tokens": int(self.condition["recall_max_tokens"]),
                "query_timestamp": self.query_timestamp,
                "trace": False,
            },
        )
        results = recall.get("results", [])
        context = "\n".join(f'- {item.get("text", "")}' for item in results) or "(no Hindsight memories retrieved)"
        response = self.client.chat(
            [
                {
                    "role": "system",
                    "content": (
                        "Answer the question using only the retrieved Hindsight memories. Resolve relative times when "
                        "a memory supplies a date. Give the best-supported brief answer, with no explanation. If "
                        "absent, say Unknown. /no_think"
                    ),
                },
                {"role": "user", "content": f"RETRIEVED MEMORIES\n{context}\n\nQUESTION\n{question}\n\nBRIEF ANSWER:"},
            ],
            seed_offset=question_index,
        )
        response["retrieval"] = {
            "bank_id": self.bank_id,
            "budget": str(self.condition["recall_budget"]),
            "max_tokens": int(self.condition["recall_max_tokens"]),
            "query_timestamp": self.query_timestamp,
            "retrieval_seconds": round(time.time() - started, 4),
            "results": results,
            "ingestion_sessions": self.ingestion_sessions,
            "ingestion_turns": self.ingestion_turns,
            "ingestion_failures": self.ingestion_failures,
            "server_health": self.health,
        }
        return response

    @staticmethod
    def _redact_secrets(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: "<redacted>"
                if any(marker in key.lower() for marker in ("api_key", "token", "password", "secret"))
                else HindsightAdapter._redact_secrets(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [HindsightAdapter._redact_secrets(item) for item in value]
        return value

    def receipt_metadata(self) -> dict[str, Any]:
        data = self.ingestion_log_path.read_bytes() if self.ingestion_log_path.exists() else b""
        raw_bank_config = self._request("GET", f"/v1/default/banks/{self.bank_id}/config") if self.bank_id else None
        bank_config = self._redact_secrets(raw_bank_config)
        llm_health = (
            self._request("POST", f"/v1/default/banks/{self.bank_id}/health/llm")
            if self.bank_id and self.version.get("features", {}).get("bank_llm_health", False)
            else {"status": "not_exposed", "feature": "bank_llm_health"}
        )
        return {
            "hindsight_repository": self.condition["hindsight_repository"],
            "hindsight_commit": self.condition["hindsight_commit"],
            "hindsight_package": self.condition["hindsight_package"],
            "server_freeze_sha256": self.condition["server_freeze_sha256"],
            "api_endpoint": self.base_url,
            "server_health": self.health,
            "server_version": self.version,
            "server_bank_config": bank_config,
            "server_llm_health": llm_health,
            "bank_ids_by_conversation_index": self.bank_ids,
            "ingestion_log_path": str(self.ingestion_log_path.relative_to(self.repo)),
            "ingestion_log_sha256": hashlib.sha256(data).hexdigest(),
            "ingestion_sessions": self.total_ingestion_sessions,
            "ingestion_turns": self.total_ingestion_turns,
            "ingestion_failures": self.total_ingestion_failures,
            "fully_local_claim": self.condition["fully_local_claim"],
        }


AX0S_RECALL_ENV_BINDINGS = {
    "recall_env_ax0s_reranker": "AX0S_RERANKER",
    "recall_env_ax0s_reranker_batch": "AX0S_RERANKER_BATCH",
    "recall_env_ax0s_rerank_fusion": "AX0S_RERANK_FUSION",
    "recall_env_ax0s_rerank_rrf_weight": "AX0S_RERANK_RRF_WEIGHT",
    "recall_env_ax0s_recall_narrow_filters": "AX0S_RECALL_NARROW_FILTERS",
}
AX0S_ENV_UNSET = "<unset>"


def bind_ax0s_recall_environment(condition: dict[str, Any]) -> dict[str, str]:
    missing = sorted(key for key in AX0S_RECALL_ENV_BINDINGS if key not in condition)
    if missing:
        raise RuntimeError(f"AX0S recall environment binding is incomplete: {missing}")
    observed = {}
    for condition_key, env_key in AX0S_RECALL_ENV_BINDINGS.items():
        expected = str(condition[condition_key])
        if expected == AX0S_ENV_UNSET:
            os.environ.pop(env_key, None)
        else:
            os.environ[env_key] = expected
        observed[env_key] = os.environ.get(env_key, AX0S_ENV_UNSET)
        if observed[env_key] != expected:
            raise RuntimeError(
                f"AX0S recall environment binding failed for {env_key}: "
                f"expected {expected!r}, observed {observed[env_key]!r}"
            )
    return observed


class Ax0sMemoryAdapter(MemoryAdapter):
    """Current ax0s-memory embedded condition over scratch-only LanceDB slugs."""

    def __init__(self, client: OpenAIChatClient, condition: dict[str, Any], repo: Path):
        import importlib.metadata

        self.recall_environment = bind_ax0s_recall_environment(condition)
        expected_root = (repo / condition["store_path"]).resolve()
        configured_root = Path(os.environ.get("AX0S_DATA_ROOT", "")).resolve()
        if configured_root != expected_root:
            raise RuntimeError(
                f"AX0S_DATA_ROOT must be {expected_root}, got {configured_root}; refusing ambiguous store"
            )
        if condition.get("reset_store", False):
            shutil.rmtree(expected_root, ignore_errors=True)
        expected_root.mkdir(parents=True, exist_ok=True)

        import subprocess

        import ax0s_memory
        from ax0s_memory import config as ax0s_config
        from ax0s_memory.connection import get_connection
        from ax0s_memory.ingest import RECORD_TABLE, upsert_rows
        from ax0s_memory.recall import recall
        from ax0s_memory.session import build_remember_row

        if ax0s_config.DATA_ROOT.resolve() != expected_root:
            raise RuntimeError(
                f"ax0s-memory resolved DATA_ROOT={ax0s_config.DATA_ROOT}, expected {expected_root}"
            )
        expected_source_path = Path(condition["source_path"]).resolve()
        observed_source_path = Path(ax0s_memory.__file__).resolve().parent.parent
        if observed_source_path != expected_source_path:
            raise RuntimeError(
                f"ax0s-memory imported from {observed_source_path}, expected {expected_source_path}"
            )
        observed_revision = subprocess.check_output(
            ["git", "-C", str(observed_source_path), "rev-parse", "HEAD"], text=True
        ).strip()
        if observed_revision != condition["source_revision"]:
            raise RuntimeError(
                f"ax0s-memory revision is {observed_revision}, expected {condition['source_revision']}"
            )
        source_status = subprocess.check_output(
            ["git", "-C", str(observed_source_path), "status", "--porcelain", "--untracked-files=all"],
            text=True,
        ).strip()
        if source_status:
            raise RuntimeError(f"ax0s-memory source checkout is dirty:\n{source_status}")
        observed_package_version = importlib.metadata.version("ax0s-memory")
        expected_values = {
            "package_version": (observed_package_version, str(condition["package_version"])),
            "embedding_model": (ax0s_config.EMBEDDING_MODEL, str(condition["embedding_model"])),
            "embedding_dims": (ax0s_config.VECTOR_DIM, int(condition["embedding_dims"])),
            "embedding_max_seq_tokens": (
                ax0s_config.MAX_SEQ_TOKENS,
                int(condition["embedding_max_seq_tokens"]),
            ),
            "recall_narrow_filters": (
                getattr(ax0s_config, "RECALL_NARROW_FILTERS", "off"),
                "off"
                if condition["recall_env_ax0s_recall_narrow_filters"] == AX0S_ENV_UNSET
                else str(condition["recall_env_ax0s_recall_narrow_filters"]),
            ),
        }
        mismatches = {
            key: {"observed": observed, "expected": expected}
            for key, (observed, expected) in expected_values.items()
            if observed != expected
        }
        if mismatches:
            raise RuntimeError(f"ax0s-memory config binding mismatch: {mismatches}")
        self.observed_source_path = observed_source_path
        self.observed_revision = observed_revision
        self.source_clean = True
        self.client = client
        self.condition = condition
        self.repo = repo
        self.store_root = expected_root
        self._get_connection = get_connection
        self._upsert_rows = upsert_rows
        self._recall = recall
        self._build_remember_row = build_remember_row
        self._record_table = RECORD_TABLE
        self.conversation_index = 0
        self.slug = ""
        self.ingestion_sessions = 0
        self.ingestion_turns = 0
        self.ingestion_failures = 0
        self.ingestion_log_path = repo / condition["ingestion_log_path"]
        self.ingestion_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.ingestion_log_path.write_text("")
        self.package_version = observed_package_version
        self.embedding_model = ax0s_config.EMBEDDING_MODEL
        self.embedding_dim = ax0s_config.VECTOR_DIM
        self.max_seq_tokens = ax0s_config.MAX_SEQ_TOKENS
        self.recall_reranker_name = (
            None if condition["recall_env_ax0s_reranker"] == AX0S_ENV_UNSET else str(condition["recall_env_ax0s_reranker"])
        )
        self.recall_rerank_fusion = (
            None
            if condition["recall_env_ax0s_rerank_fusion"] == AX0S_ENV_UNSET
            else str(condition["recall_env_ax0s_rerank_fusion"])
        )
        self.recall_rerank_rrf_weight = (
            None
            if condition["recall_env_ax0s_rerank_rrf_weight"] == AX0S_ENV_UNSET
            else float(condition["recall_env_ax0s_rerank_rrf_weight"])
        )

    @staticmethod
    def _utc_z(value: datetime) -> str:
        return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")

    @staticmethod
    def _retrieved_at() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")

    async def _count_records(self) -> int:
        last_error: Exception | None = None
        for _ in range(2):
            try:
                conn = await self._get_connection(self.slug)
                table = await conn.open_table(self._record_table)
                return int(await table.count_rows())
            except Exception as exc:
                last_error = exc
        assert last_error is not None
        raise last_error

    async def _ingest(self, conversation: dict[str, Any], conversation_index: int) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        log_rows: list[dict[str, Any]] = []
        speaker_a = str(conversation["speaker_a"])
        sessions = chronological_sessions(conversation)
        retrieved_at = self._retrieved_at()
        for number, date, turns in sessions:
            parsed = parse_locomo_datetime(date)
            if parsed is None:
                raise RuntimeError(f"unparseable LoCoMo session timestamp: {date!r}")
            self.ingestion_sessions += 1
            for turn_index, turn in enumerate(turns):
                text = locomo_turn_text(turn)
                if not text:
                    continue
                speaker = str(turn.get("speaker", ""))
                role = "user" if speaker == speaker_a else "assistant"
                ts = self._utc_z(parsed + timedelta(seconds=turn_index))
                source_id = f"locomo:{conversation_index}:{number}:{turn_index}"
                rid = "record:r_" + hashlib.sha256(source_id.encode()).hexdigest()[:12]
                content = f"[Authoritative message timestamp: {date}]\n{speaker}: {text}"
                temporal = {
                    "occurred_at": ts,
                    "retrieved_at": retrieved_at,
                    "synced_at": None,
                    "synced_at_reason": "embedded benchmark input; no upstream sync clock exists",
                    "tz": "UTC",
                    "tz_offset_minutes": 0,
                    "offset_source": "native",
                    "tz_source": "pinned harness convention: LoCoMo naive timestamp interpreted as UTC",
                    "temporal_precision": "minute",
                }
                row, _, _, _ = await self._build_remember_row(
                    self.slug,
                    text_content=content,
                    role=role,
                    record_type="conversation",
                    tags=["locomo", "benchmark", f"session-{number}"],
                    ts=ts,
                    record_id=rid,
                    extras={
                        "agent": "benchmark-runner",
                        "entrypoint": "benchmark",
                        "source_type": "prime",
                        "source_id": source_id,
                        "temporal": temporal,
                    },
                )
                rows.append(row)
                log_rows.append(
                    {
                        "conversation_index": conversation_index,
                        "slug": self.slug,
                        "session": number,
                        "session_date": date,
                        "turn_index": turn_index,
                        "speaker": speaker,
                        "role": role,
                        "record_id": rid,
                        "source_id": source_id,
                        "ts": ts,
                        "input_sha256": hashlib.sha256(content.encode()).hexdigest(),
                        "error": None,
                    }
                )
        started = time.monotonic()
        result = await self._upsert_rows(self.slug, {self._record_table: rows})
        elapsed = time.monotonic() - started
        count = await self._count_records()
        if count != len(rows):
            raise RuntimeError(
                f"ax0s-memory ingestion count mismatch for {self.slug}: stored={count}, expected={len(rows)}"
            )
        for item in log_rows:
            item["batch_ingestion_seconds"] = round(elapsed, 4)
            item["stored_record_count_after_batch"] = count
            with self.ingestion_log_path.open("a") as log:
                log.write(json.dumps(item, ensure_ascii=False) + "\n")
        return {
            "rows": len(rows),
            "stored_record_count": count,
            "ingest_result_counts": result.counts,
            "ingestion_seconds": round(elapsed, 4),
        }

    def ingest_conversation(self, conversation: dict[str, Any]) -> None:
        conversation_index = self.conversation_index
        self.slug = f'{self.condition["slug_prefix"]}-{conversation_index}'
        self.conversation_index += 1
        self.ingestion_sessions = 0
        self.ingestion_turns = 0
        try:
            summary = asyncio.run(self._ingest(conversation, conversation_index))
        except Exception:
            self.ingestion_failures += 1
            raise
        self.ingestion_turns = int(summary["rows"])
        print(
            f"ax0s ingestion: slug={self.slug} rows={summary['rows']} "
            f"stored={summary['stored_record_count']} seconds={summary['ingestion_seconds']}",
            flush=True,
        )

    def answer(self, question: str, *, question_index: int) -> dict[str, Any]:
        if not self.slug:
            raise RuntimeError("ingest_conversation must be called before answer")
        self.recall_environment = bind_ax0s_recall_environment(self.condition)
        started = time.monotonic()
        result = asyncio.run(
            self._recall(
                self.slug,
                question,
                mode=str(self.condition.get("mode", "hybrid")),
                limit=int(self.condition["top_k"]),
                expand_incidence=False,
                min_score=self.condition.get("min_score"),
                ranker=str(self.condition.get("ranker", "weighted")),
                reranker_name=self.recall_reranker_name,
                rerank_fusion=self.recall_rerank_fusion,
                rerank_rrf_weight=self.recall_rerank_rrf_weight,
                dedup_hits=False,
            )
        )
        retrieval_seconds = time.monotonic() - started
        retrieval = result.to_json()
        hits = retrieval.get("hits", [])
        context = "\n".join(f'- {item["text_content"]}' for item in hits) or "(no memories retrieved)"
        response = self.client.chat(
            [
                {
                    "role": "system",
                    "content": (
                        "Answer the question using only the retrieved AX0S Memory records. Resolve relative times "
                        "when a record supplies a date. Give the best-supported brief answer, with no explanation. "
                        "If absent, say Unknown. /no_think"
                    ),
                },
                {
                    "role": "user",
                    "content": f"RETRIEVED RECORDS\n{context}\n\nQUESTION\n{question}\n\nBRIEF ANSWER:",
                },
            ],
            seed_offset=question_index,
        )
        response["retrieval"] = {
            "slug": self.slug,
            "mode": self.condition.get("mode", "hybrid"),
            "ranker": self.condition.get("ranker", "weighted"),
            "top_k": int(self.condition["top_k"]),
            "min_score": self.condition.get("min_score"),
            "retrieval_seconds": round(retrieval_seconds, 4),
            "hits": hits,
            "ingestion_sessions": self.ingestion_sessions,
            "ingestion_turns": self.ingestion_turns,
            "ingestion_failures": self.ingestion_failures,
        }
        return response

    def receipt_metadata(self) -> dict[str, Any]:
        data = self.ingestion_log_path.read_bytes()
        rows = [json.loads(line) for line in data.decode().splitlines() if line]
        by_conversation: dict[str, int] = {}
        for row in rows:
            key = str(row["conversation_index"])
            by_conversation[key] = by_conversation.get(key, 0) + 1
        return {
            "ax0s_memory": {
                "package_version": self.package_version,
                "source_revision": self.condition.get("source_revision"),
                "source_path": self.condition.get("source_path"),
                "observed_source_revision": self.observed_revision,
                "observed_import_source_path": str(self.observed_source_path),
                "observed_source_clean": self.source_clean,
                "embedding_model": self.embedding_model,
                "embedding_dim": self.embedding_dim,
                "max_seq_tokens": self.max_seq_tokens,
                "embedder_device_policy": "CPU forced by CUDA_VISIBLE_DEVICES=''",
                "store_root": str(self.store_root),
                "scratch_only": True,
                "recall_environment_binding": self.recall_environment,
                "recall_call_binding": {
                    "reranker_name": self.recall_reranker_name,
                    "rerank_fusion": self.recall_rerank_fusion,
                    "rerank_rrf_weight": self.recall_rerank_rrf_weight,
                },
            },
            "ingestion": {
                "log_path": str(self.ingestion_log_path.relative_to(self.repo)),
                "log_sha256": hashlib.sha256(data).hexdigest(),
                "turns": len(rows),
                "failures": sum(int(bool(row.get("error"))) for row in rows),
                "by_conversation": by_conversation,
                "complete": bool(rows) and not any(row.get("error") for row in rows),
            },
        }

def make_adapter(
    name: str, client: OpenAIChatClient, condition: dict[str, Any], repo: Path
) -> MemoryAdapter:
    if name == "full_context":
        return FullContextAdapter(client)
    if name == "ax0s_memory":
        return Ax0sMemoryAdapter(client, condition, repo)
    if name == "mem0":
        return Mem0Adapter(client, condition, repo)
    if name == "mem0_turn":
        return Mem0TurnAdapter(client, condition, repo)
    if name == "memmachine":
        return MemMachineAdapter(client, condition, repo)
    if name == "graphiti":
        return GraphitiAdapter(client, condition, repo)
    if name == "hindsight":
        return HindsightAdapter(client, condition, repo)
    raise ValueError(f"unknown adapter: {name}")
