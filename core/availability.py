"""Can this environment actually run each CLI?

Two different questions, and the UI needs both. *Installed* is cheap and local:
is the binary on PATH. *Authenticated* is neither — it means asking the CLI,
which costs a subprocess — so it is reported separately and only when asked
for, and a slow or hung probe reports "unknown" rather than blocking a picker.
"""

from __future__ import annotations

import subprocess
from dataclasses import asdict, dataclass

from cliagents import Surface, registry

#: How the CLI is asked whether it has credentials. These are read-only status
#: commands: no model call, no tokens, no side effects.
_AUTH_PROBES = {
    "claude-code": ["auth", "status"],
    "codex": ["login", "status"],
}

_PROBE_TIMEOUT_SECONDS = 10


@dataclass(frozen=True)
class CliAvailability:
    agent_id: str
    display_name: str
    installed: bool
    executable_path: str | None
    install_hint: str
    #: True / False when the probe answered, None when it was not run or could
    #: not be trusted. A picker shows the difference; guessing False here would
    #: hide a working agent behind a "log in first" that is not true.
    authenticated: bool | None
    auth_hint: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


def snapshot(*, refresh: bool = True, check_auth: bool = False) -> list[CliAvailability]:
    """Availability for every agent that can be given work.

    Filtered by the exec surface, not by the terminal one: this list drives
    "who can I delegate to", and an agent that only knows how to hold an
    interactive session does not belong in it.
    """
    out: list[CliAvailability] = []
    for provider in registry.for_surface(Surface.EXEC):
        detected = provider.detect(refresh=refresh)
        authenticated: bool | None = None
        auth_hint = ""
        if check_auth and detected.installed:
            authenticated, auth_hint = _probe_auth(provider.id, provider.executable)
        out.append(
            CliAvailability(
                agent_id=provider.id,
                display_name=provider.display_name,
                installed=detected.installed,
                executable_path=detected.executable_path,
                install_hint=provider.install_hint,
                authenticated=authenticated,
                auth_hint=auth_hint,
            )
        )
    return out


def _probe_auth(agent_id: str, executable: str, *, runner=subprocess.run) -> tuple[bool | None, str]:
    probe = _AUTH_PROBES.get(agent_id)
    if not probe:
        # An agent we have no probe for is not an agent we get to call
        # unauthenticated.
        return None, ""
    try:
        completed = runner(
            [executable, *probe],
            capture_output=True,
            text=True,
            timeout=_PROBE_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError):
        return None, ""

    if completed.returncode == 0:
        return True, ""
    detail = (completed.stderr or completed.stdout or "").strip().splitlines()
    return False, detail[0] if detail else "not logged in"
