"""Shared test isolation.

Every run asks the database for its tenant's API key (``core.credentials``).
No test should reach a real database for that because the developer's shell
happens to have ``SUPABASE_URL`` set, and the per-process cache must not carry
one test's answer into the next. Tests that exercise the lookup replace
``_fetch`` themselves.
"""

from __future__ import annotations

import pytest

from core import credentials


@pytest.fixture(autouse=True)
def _no_tenant_key_lookup(monkeypatch):
    credentials.clear_cache()
    monkeypatch.setattr(credentials, "_fetch", lambda _tenant, _provider: None)
    yield
    credentials.clear_cache()
