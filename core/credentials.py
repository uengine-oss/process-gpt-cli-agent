"""Running each CLI on its tenant's own API key.

Until now every run used one Claude subscription token, mounted into the pod.
Anthropic's terms do not allow a subscription credential to be shared by many
people or used to serve other people's requests; a company's own API key, used
by that company's authorised users, is allowed. So a tenant administrator
stores the company's Anthropic and OpenAI keys (Supabase Vault, behind
``cli_agent_credentials``), and each run reads its tenant's key here and hands
it to that run's CLI process.

Three things this module guarantees, because each failure is quiet:

* **The key reaches one process.** It goes into the environment built for
  the one CLI child of that tenant's run — never ``os.environ``, which every
  concurrent run (another tenant's included) would inherit.
* **The key replaces the pod's credentials; it does not merely sit beside
  them.** With both present, which one is billed depends on the CLI's
  precedence rules and their version, and billing a tenant's work to the
  shared subscription is exactly what this exists to stop. So the competing
  variables are removed from that run's environment.
* **No key means today's behaviour.** A tenant that has not set one — or a
  database that does not have the function yet — runs on the pod's
  credentials, and the run's record says so.

The key is never written anywhere by this service: not to a log, an event, the
timeline or the session archive. :func:`redact` and :func:`scrub_workspace`
cover the other direction — the agent itself printing it.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .settings import settings

logger = logging.getLogger(__name__)

#: The service-role-only function that decrypts a tenant's key from Vault.
RPC = "cli_agent_credential_get"

#: What a matched secret is replaced with wherever the agent echoed it.
REDACTED = "[redacted]"

#: Error codes that mean "the credential store is not there", as opposed to
#: "the lookup failed": PostgREST's unknown function, and Postgres' undefined
#: function / table / schema (the function exists but Vault or the table does
#: not). Production runs before the SQL is applied, and that is not an error.
_MISSING_CODES = frozenset({"PGRST202", "42883", "42P01", "3F000"})

#: Files larger than this are not scanned for an echoed key. Transcripts and
#: text outputs are far smaller; a large file is a binary deliverable, where
#: a key would not appear as plain text anyway.
_SCRUB_MAX_BYTES = 50 * 1024 * 1024


@dataclass(frozen=True)
class _Plan:
    """How one CLI is pointed at a tenant key."""

    #: ``cli_agent_credentials.provider``.
    provider: str
    #: Shown in the run's record. Never the key.
    label: str
    #: Variables set to the key.
    variables: tuple[str, ...]
    #: Pod credentials removed from the run, because the CLI would prefer them
    #: to the key or because which one wins is not something to leave to a
    #: CLI version.
    replaces: tuple[str, ...] = ()


_PLANS: dict[str, _Plan] = {
    # Claude Code's order is ANTHROPIC_AUTH_TOKEN, then ANTHROPIC_API_KEY,
    # then apiKeyHelper, then CLAUDE_CODE_OAUTH_TOKEN. The bearer token would
    # outrank the tenant key outright; the subscription token ranks below it
    # today, but the deployment manifest has seen that differ by version.
    "claude-code": _Plan(
        provider="anthropic",
        label="Anthropic",
        variables=("ANTHROPIC_API_KEY",),
        replaces=("CLAUDE_CODE_OAUTH_TOKEN", "ANTHROPIC_AUTH_TOKEN"),
    ),
    # `codex exec` reads CODEX_API_KEY; older releases read OPENAI_API_KEY.
    # Set both so neither falls back to the pod's. Its login file lives under
    # CODEX_HOME, which each run already relocates into its own workspace.
    "codex": _Plan(
        provider="openai",
        label="OpenAI",
        variables=("CODEX_API_KEY", "OPENAI_API_KEY"),
    ),
}


class _StoreUnavailableError(Exception):
    """The credential store does not exist in this database (yet)."""


class KeyLookupError(Exception):
    """The tenant's key could not be read. ``reason`` says why, for the record."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class RunCredentials:
    """Which credentials one run's CLI process is started with.

    ``env`` holds the key and is left out of ``repr``, so a stray
    ``logger.debug("%r", creds)`` does not print it.
    """

    agent_id: str
    tenant_id: str
    #: ``anthropic`` / ``openai``; empty for a CLI without tenant keys.
    provider: str = ""
    label: str = ""
    env: Mapping[str, str] = field(default_factory=dict, repr=False)
    unset: frozenset[str] = frozenset()
    #: Why the pod's credentials are used: ``not_set``, ``unavailable``,
    #: ``lookup_failed``, ``no_tenant``, ``unsupported``. Empty with a key.
    reason: str = ""

    @property
    def tenant_key(self) -> bool:
        return bool(self.env)

    @property
    def secrets(self) -> tuple[str, ...]:
        """The values that must not appear in anything this run leaves behind."""
        return tuple(sorted({v for v in self.env.values() if v}))

    def apply(self, env: Mapping[str, str]) -> dict[str, str]:
        """``env`` with the replaced credentials removed and the key set."""
        out = {k: v for k, v in env.items() if k not in self.unset}
        out.update(self.env)
        return out

    def bind(self, provider):
        """The provider to start this run with.

        ``Provider.exec_env(base)`` starts from ``os.environ`` and only adds to
        it — the library has no way to leave a variable out, and setting one
        to an empty string is not the same as unsetting it to every CLI. So
        with a tenant key the provider is wrapped, and ``exec_env``, the one
        method ``stream_exec`` asks for the child's environment, removes the
        replaced variables after that merge. Without a key the provider is
        returned unchanged: today's path, byte for byte.
        """
        if not self.tenant_key:
            return provider
        return _TenantKeyProvider(provider, self)

    def describe(self) -> str:
        """One line for the run's record. Never contains the key."""
        if self.tenant_key:
            return f"회사 {self.label} API 키로 실행합니다."
        fallback = "이 서버의 기본 자격 증명으로 실행합니다."
        if self.reason == "not_set":
            return f"회사 {self.label} API 키가 설정되지 않아 {fallback}"
        if self.reason in ("unavailable", "lookup_failed"):
            return f"회사 {self.label} API 키를 확인하지 못해 {fallback}"
        if self.reason == "no_tenant":
            return f"테넌트를 알 수 없어 회사 API 키 없이 {fallback}"
        return f"이 CLI 는 회사 API 키를 지원하지 않아 {fallback}"


class _TenantKeyProvider:
    """A provider whose headless runs see the tenant key and not the pod's.

    Delegates everything else, so the run is otherwise the same run.
    """

    def __init__(self, provider, credentials: RunCredentials) -> None:
        self._provider = provider
        self._credentials = credentials

    def __getattr__(self, name: str) -> Any:
        # Only reached for names this wrapper does not define. Guarded so a
        # copy or pickle, which looks attributes up before __init__, does not
        # recurse into itself.
        if name in ("_provider", "_credentials"):
            raise AttributeError(name)
        return getattr(self._provider, name)

    def exec_env(self, base: dict[str, str] | None = None) -> dict[str, str]:
        return self._credentials.apply(self._provider.exec_env(base))

    def __repr__(self) -> str:
        return f"<{getattr(self._provider, 'id', '?')} with the tenant's key>"


# -- lookup ------------------------------------------------------------------

#: A cached "the store is not there", kept apart from "no key" so a cached
#: answer is described the same way as the first one.
_UNAVAILABLE = object()

_cache: dict[tuple[str, str], tuple[float, object]] = {}
_cache_lock = threading.Lock()
_reported_unavailable = False


def clear_cache() -> None:
    """Forget every cached answer, and log a missing store again."""
    global _reported_unavailable
    with _cache_lock:
        _cache.clear()
        _reported_unavailable = False


def _fetch(tenant_id: str, provider: str) -> str | None:
    """Ask the database for one tenant's key, with the service-role client.

    The same client the rest of the service uses (the SDK's, built from
    ``SUPABASE_URL`` / ``SUPABASE_KEY``), so there is one credential to
    configure. ``cli_agent_credential_get`` is granted to service_role only.
    """
    if not (os.getenv("SUPABASE_URL") or os.getenv("SUPABASE_KEY_URL")):
        # Not configured at all (a local run, a test). Asking would only log
        # the SDK's connection error on every run.
        raise _StoreUnavailableError("SUPABASE_URL is not set")

    from processgpt_agent_sdk.database import get_db_client, initialize_db

    initialize_db()
    try:
        response = (
            get_db_client().rpc(RPC, {"p_tenant": tenant_id, "p_provider": provider}).execute()
        )
    except Exception as exc:
        if str(getattr(exc, "code", "") or "") in _MISSING_CODES:
            raise _StoreUnavailableError(
                f"{getattr(exc, 'code', '')} {getattr(exc, 'message', '')}"
            ) from exc
        raise
    value = getattr(response, "data", None)
    if isinstance(value, str) and value.strip():
        # Pasted keys come with a trailing newline often enough to matter, and
        # a key with one fails authentication with an unhelpful message.
        return value.strip()
    return None


def lookup(
    tenant_id: str,
    provider: str,
    *,
    clock: Callable[[], float] = time.monotonic,
) -> str | None:
    """A tenant's key for ``provider``, or None. Cached for a short while.

    Misses are cached too: a tenant without a key would otherwise cost a round
    trip on every run. A key saved or removed in the admin screen therefore
    takes effect within ``CLIAGENTS_CREDENTIAL_CACHE_SECONDS``.

    A store that does not exist yet is a cached miss, logged once per process.
    Any other failure is not cached — the next run asks again. Both raise
    :class:`KeyLookupError`, so the caller can say the key could not be checked.
    """
    global _reported_unavailable
    key = (tenant_id, provider)
    now = clock()
    with _cache_lock:
        hit = _cache.get(key)
    if hit is not None and hit[0] > now:
        if hit[1] is _UNAVAILABLE:
            raise KeyLookupError("unavailable")
        return hit[1]  # type: ignore[return-value]

    ttl = max(0, settings.credential_cache_seconds)
    try:
        value = _fetch(tenant_id, provider)
    except _StoreUnavailableError as exc:
        with _cache_lock:
            first = not _reported_unavailable
            _reported_unavailable = True
            _cache[key] = (now + ttl, _UNAVAILABLE)
        if first:
            logger.info(
                "tenant API keys unavailable (%s); runs use the pod's credentials until %s exists",
                exc,
                RPC,
            )
        raise KeyLookupError("unavailable") from None
    except Exception as exc:  # noqa: BLE001 - reported by the caller, run continues
        # The exception carries the request (tenant, provider), never a key:
        # the key only exists in a successful response.
        logger.warning(
            "tenant API key lookup failed | tenant=%s provider=%s error=%s: %s",
            tenant_id,
            provider,
            type(exc).__name__,
            exc,
        )
        raise KeyLookupError("lookup_failed") from None

    with _cache_lock:
        _cache[key] = (now + ttl, value)
    return value


def for_run(agent_id: str, tenant_id: str) -> RunCredentials:
    """The credentials a run of ``agent_id`` for ``tenant_id`` starts with.

    Blocking (a database round trip on a cache miss); call it off the loop.
    Never raises: every failure means "the pod's credentials", stated in
    ``reason``.
    """
    tenant_id = (tenant_id or "").strip()
    plan = _PLANS.get(agent_id)
    if plan is None:
        return RunCredentials(agent_id=agent_id, tenant_id=tenant_id, reason="unsupported")
    base = RunCredentials(
        agent_id=agent_id, tenant_id=tenant_id, provider=plan.provider, label=plan.label
    )
    if not tenant_id:
        return dataclasses.replace(base, reason="no_tenant")

    try:
        secret = lookup(tenant_id, plan.provider)
    except KeyLookupError as exc:
        return dataclasses.replace(base, reason=exc.reason)
    if not secret:
        return dataclasses.replace(base, reason="not_set")

    return dataclasses.replace(
        base,
        env={name: secret for name in plan.variables},
        unset=frozenset(plan.replaces),
    )


# -- keeping it out of what the run leaves behind ------------------------------


def _scrub(value: Any, secrets: tuple[str, ...]) -> Any:
    if isinstance(value, str):
        for secret in secrets:
            if secret in value:
                value = value.replace(secret, REDACTED)
        return value
    if isinstance(value, dict):
        return {k: _scrub(v, secrets) for k, v in value.items()}
    if isinstance(value, list):
        return [_scrub(v, secrets) for v in value]
    if isinstance(value, tuple):
        return tuple(_scrub(v, secrets) for v in value)
    return value


def redact(event, secrets: Iterable[str]):
    """``event`` with every occurrence of a secret replaced.

    The service never puts the key in an event; the agent can — a shell
    command that prints the environment, a file it was told to read. Every
    event feeds the chat stream, the timeline, the journal and the failure
    text, so cleaning it once here covers all of them.
    """
    secrets = tuple(s for s in secrets if s)
    if not secrets:
        return event
    changes = {}
    for name in ("text", "tool_input", "tool_output", "path", "raw"):
        before = getattr(event, name)
        after = _scrub(before, secrets)
        if after != before:
            changes[name] = after
    return dataclasses.replace(event, **changes) if changes else event


def scrub_workspace(root: Path, secrets: Iterable[str]) -> list[Path]:
    """Replace an echoed key in the files a run left in its workspace.

    The CLI's session transcript records tool output verbatim, and the
    workspace is what the session archive uploads and what downloads read
    from. Run after the CLI has exited and before any of that happens.
    Returns the files changed.
    """
    needles = [(s.encode(), REDACTED.encode()) for s in secrets if s]
    if not needles or not root.is_dir():
        return []
    changed: list[Path] = []
    for path in root.rglob("*"):
        try:
            if path.is_symlink() or not path.is_file():
                continue
            if path.stat().st_size > _SCRUB_MAX_BYTES:
                continue
            data = path.read_bytes()
            cleaned = data
            for needle, replacement in needles:
                cleaned = cleaned.replace(needle, replacement)
            if cleaned != data:
                path.write_bytes(cleaned)
                changed.append(path)
        except OSError:
            continue
    if changed:
        # Paths only: the point is that the content is gone.
        logger.warning(
            "removed an echoed tenant API key from %d workspace file(s) under %s",
            len(changed),
            root,
        )
    return changed
