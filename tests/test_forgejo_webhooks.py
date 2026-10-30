"""Tests for the Forgejo webhook route + author filter dispatch."""

from __future__ import annotations

import hashlib
import hmac
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from mira.config import FilterConfig, MiraConfig
from mira.platforms.forgejo.auth import ForgejoTokenAuth
from mira.platforms.server import create_app

FJ_SECRET = "fj-secret"
BOT = "mira-bot"


@pytest.fixture
def forgejo_auth():  # noqa: ANN201
    auth = ForgejoTokenAuth("tok")
    auth.get_bot_identity = AsyncMock(return_value="mira-bot")  # type: ignore[method-assign]
    return auth


@pytest.fixture
def app(forgejo_auth):  # noqa: ANN001, ANN201
    return create_app(
        app_auth=None,
        webhook_secret=None,
        bot_name=BOT,
        forgejo_auth=forgejo_auth,
        forgejo_webhook_secret=FJ_SECRET,
    )


@pytest.fixture
async def client(app) -> AsyncClient:  # noqa: ANN001
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


def _sign(payload_bytes: bytes) -> str:
    return hmac.new(FJ_SECRET.encode(), payload_bytes, hashlib.sha256).hexdigest()


def _pr_payload(action: str, login: str):
    return {
        "action": action,
        "pull_request": {"number": 7},
        "repository": {"full_name": "o/r", "private": False},
        "sender": {"login": login},
    }


def _comment_payload(body: str, login: str):
    return {
        "action": "created",
        "is_pull": True,
        "comment": {"body": body},
        "sender": {"login": login},
    }


@pytest.mark.asyncio
async def test_pr_opened_blocked_author_filtered(client):
    """Blocked author with [bot] suffix is filtered for PR opened."""
    with (
        patch("mira.platforms.forgejo.webhook.handle_forgejo_pr", new=AsyncMock()) as h,
        patch(
            "mira.platforms.forgejo.webhook.load_config",
            return_value=MiraConfig(filter=FilterConfig(blocked_authors=["dependabot"])),
        ),
    ):
        payload = _pr_payload("opened", "dependabot[bot]")
        body = json.dumps(payload).encode()
        resp = await client.post(
            "/forgejo/webhook",
            content=body,
            headers={
                "X-Forgejo-Event": "pull_request",
                "X-Forgejo-Signature": _sign(body),
            },
        )
    assert resp.status_code == 200
    assert resp.json()["status"] == "ignored"
    h.assert_not_called()


@pytest.mark.asyncio
async def test_pr_opened_allowed_author_not_filtered(client):
    """Non-blocked author passes through the filter."""
    with (
        patch("mira.platforms.forgejo.webhook.handle_forgejo_pr", new=AsyncMock()) as h,
        patch(
            "mira.platforms.forgejo.webhook.load_config",
            return_value=MiraConfig(filter=FilterConfig(blocked_authors=["dependabot"])),
        ),
    ):
        payload = _pr_payload("opened", "alice")
        body = json.dumps(payload).encode()
        resp = await client.post(
            "/forgejo/webhook",
            content=body,
            headers={
                "X-Forgejo-Event": "pull_request",
                "X-Forgejo-Signature": _sign(body),
            },
        )
    assert resp.status_code == 200
    assert resp.json()["status"] == "processing"
    h.assert_awaited_once()


@pytest.mark.asyncio
async def test_pr_opened_allowlist_filters_off_list(client):
    """Allowlist set but author not on it → filtered."""
    with (
        patch("mira.platforms.forgejo.webhook.handle_forgejo_pr", new=AsyncMock()) as h,
        patch(
            "mira.platforms.forgejo.webhook.load_config",
            return_value=MiraConfig(filter=FilterConfig(allowed_authors=["alice"])),
        ),
    ):
        payload = _pr_payload("opened", "bob")
        body = json.dumps(payload).encode()
        resp = await client.post(
            "/forgejo/webhook",
            content=body,
            headers={
                "X-Forgejo-Event": "pull_request",
                "X-Forgejo-Signature": _sign(body),
            },
        )
    assert resp.status_code == 200
    assert resp.json()["status"] == "ignored"
    h.assert_not_called()


@pytest.mark.asyncio
async def test_comment_review_ignores_author_filter(client):
    """Commands use the command policy, not the auto-review author filter."""
    with (
        patch("mira.platforms.forgejo.webhook.handle_forgejo_note", new=AsyncMock()) as h,
        patch(
            "mira.platforms.forgejo.webhook.load_config",
            return_value=MiraConfig(filter=FilterConfig(blocked_authors=["dependabot"])),
        ),
    ):
        payload = _comment_payload("@mira-bot review", "dependabot[bot]")
        body = json.dumps(payload).encode()
        resp = await client.post(
            "/forgejo/webhook",
            content=body,
            headers={
                "X-Forgejo-Event": "issue_comment",
                "X-Forgejo-Signature": _sign(body),
            },
        )
    assert resp.status_code == 200
    assert resp.json()["status"] == "processing"
    h.assert_awaited_once()


@pytest.mark.asyncio
async def test_comment_non_review_ignores_author_filter(client):
    """Free-form command dispatch is separate from the auto-review author filter."""
    with (
        patch("mira.platforms.forgejo.webhook.handle_forgejo_note", new=AsyncMock()) as h,
        patch(
            "mira.platforms.forgejo.webhook.load_config",
            return_value=MiraConfig(filter=FilterConfig(blocked_authors=["alice"])),
        ),
    ):
        payload = _comment_payload("@mira-bot pause", "alice")
        body = json.dumps(payload).encode()
        resp = await client.post(
            "/forgejo/webhook",
            content=body,
            headers={
                "X-Forgejo-Event": "issue_comment",
                "X-Forgejo-Signature": _sign(body),
            },
        )
    assert resp.status_code == 200
    assert resp.json()["status"] == "processing"
    h.assert_awaited_once()


# ── Merge-time learning (pull_request / action=closed + merged) ──


def _closed_pr_payload(
    *,
    merged: bool | None = True,
    merged_by: str | None = "carol",
    login: str = "alice",
    full_name: str = "o/r",
):
    pr: dict = {
        "number": 7,
        "html_url": f"https://forgejo.example/{full_name}/pulls/7",
        "title": "Add feature",
        "body": "",
    }
    if merged is not None:
        pr["merged"] = merged
    if merged_by is not None:
        pr["merged_by"] = {"login": merged_by}
    return {
        "action": "closed",
        "pull_request": pr,
        "repository": {
            "full_name": full_name,
            "private": False,
            "html_url": f"https://forgejo.example/{full_name}",
        },
        "sender": {"login": login},
    }


async def _post_pr(client, payload):
    body = json.dumps(payload).encode()
    return await client.post(
        "/forgejo/webhook",
        content=body,
        headers={
            "X-Forgejo-Event": "pull_request",
            "X-Forgejo-Signature": _sign(body),
        },
    )


@pytest.mark.asyncio
async def test_pr_closed_merged_dispatches_merge_learning(client):
    """A signed closed+merged PR is dispatched to merge-time learning."""
    with (
        patch("mira.platforms.forgejo.webhook.handle_forgejo_merge", new=AsyncMock()) as h,
        patch("mira.platforms.forgejo.webhook.load_config", return_value=MiraConfig()),
    ):
        resp = await _post_pr(client, _closed_pr_payload(merged=True))
    assert resp.status_code == 200
    assert resp.json()["status"] == "processing"
    h.assert_awaited_once()


@pytest.mark.asyncio
async def test_pr_closed_unmerged_ignored(client):
    """A closed-but-not-merged PR must not trigger merge-time learning."""
    with (
        patch("mira.platforms.forgejo.webhook.handle_forgejo_merge", new=AsyncMock()) as h,
        patch("mira.platforms.forgejo.webhook.load_config", return_value=MiraConfig()),
    ):
        resp = await _post_pr(client, _closed_pr_payload(merged=False))
    assert resp.status_code == 200
    assert resp.json()["status"] == "ignored"
    h.assert_not_called()


@pytest.mark.asyncio
async def test_pr_closed_missing_merged_field_ignored(client):
    """Missing pull_request.merged is treated as not merged."""
    with (
        patch("mira.platforms.forgejo.webhook.handle_forgejo_merge", new=AsyncMock()) as h,
        patch("mira.platforms.forgejo.webhook.load_config", return_value=MiraConfig()),
    ):
        resp = await _post_pr(client, _closed_pr_payload(merged=None))
    assert resp.status_code == 200
    assert resp.json()["status"] == "ignored"
    h.assert_not_called()


@pytest.mark.asyncio
async def test_pr_closed_merged_self_event_ignored(client):
    """Bot-authored closed events are skipped by the self-event filter."""
    with (
        patch("mira.platforms.forgejo.webhook.handle_forgejo_merge", new=AsyncMock()) as h,
        patch("mira.platforms.forgejo.webhook.load_config", return_value=MiraConfig()),
    ):
        resp = await _post_pr(client, _closed_pr_payload(login=BOT))
    assert resp.status_code == 200
    assert resp.json()["status"] == "ignored"
    h.assert_not_called()


@pytest.mark.asyncio
async def test_pr_closed_malformed_payloads_are_sane(client):
    """Empty/structurally-broken payloads are ignored, never 500."""
    with (
        patch("mira.platforms.forgejo.webhook.handle_forgejo_merge", new=AsyncMock()) as h,
        patch("mira.platforms.forgejo.webhook.load_config", return_value=MiraConfig()),
    ):
        for payload in (
            {},  # completely empty
            {"action": "closed"},  # no pull_request, repository or sender
            {
                "action": "closed",
                "pull_request": {},
                "repository": {"full_name": "broken"},
                "sender": {"login": "alice"},
            },  # empty PR + malformed full_name
        ):
            resp = await _post_pr(client, payload)
            assert resp.status_code == 200
            assert resp.json()["status"] == "ignored"
    h.assert_not_called()


@pytest.mark.asyncio
async def test_handle_forgejo_merge_uses_forgejo_platform_and_merged_by(forgejo_auth):
    """The handler runs the shared learning with platform='forgejo' and the
    identity from pull_request.merged_by."""
    from mira.models import PRInfo
    from mira.platforms.forgejo import webhook as fw

    pr_info = PRInfo(
        title="Add feature",
        description="",
        base_branch="main",
        head_branch="f",
        url="https://forgejo.example/o/r/pulls/7",
        number=7,
        owner="o",
        repo="r",
        platform="forgejo",
        author="dave",
    )
    provider = MagicMock()
    provider.get_pr_info = AsyncMock(return_value=pr_info)
    with (
        patch("mira.platforms.forgejo.webhook.create_provider", return_value=provider) as cp,
        patch("mira.platforms.handlers.run_pr_merged_learning", new=AsyncMock()) as learn,
    ):
        await fw.handle_forgejo_merge(
            _closed_pr_payload(merged=True, merged_by="carol"), forgejo_auth, BOT
        )
    cp.assert_called_once_with("forgejo", "tok")
    learn.assert_awaited_once()
    args, kwargs = learn.await_args
    assert args[1] is pr_info
    assert args[2] == BOT
    assert args[3] == "carol"
    assert kwargs.get("platform") == "forgejo"


@pytest.mark.asyncio
async def test_handle_forgejo_merge_identity_fallbacks(forgejo_auth):
    """merged_by falls back to its username key, then to the sender login."""
    from mira.models import PRInfo
    from mira.platforms.forgejo import webhook as fw

    pr_info = PRInfo(
        title="",
        description="",
        base_branch="main",
        head_branch="f",
        url="https://forgejo.example/o/r/pulls/7",
        number=7,
        owner="o",
        repo="r",
        platform="forgejo",
        author="dave",
    )
    provider = MagicMock()
    provider.get_pr_info = AsyncMock(return_value=pr_info)

    payload = _closed_pr_payload(merged=True, merged_by=None, login="alice")
    payload["pull_request"]["merged_by"] = {"username": "carol"}
    with (
        patch("mira.platforms.forgejo.webhook.create_provider", return_value=provider),
        patch("mira.platforms.handlers.run_pr_merged_learning", new=AsyncMock()) as learn,
    ):
        await fw.handle_forgejo_merge(payload, forgejo_auth, BOT)
    assert learn.await_args.args[3] == "carol"

    learn.reset_mock()
    payload = _closed_pr_payload(merged=True, merged_by=None, login="alice")
    with (
        patch("mira.platforms.forgejo.webhook.create_provider", return_value=provider),
        patch("mira.platforms.handlers.run_pr_merged_learning", new=AsyncMock()) as learn,
    ):
        await fw.handle_forgejo_merge(payload, forgejo_auth, BOT)
    assert learn.await_args.args[3] == "alice"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        {},  # completely empty
        {"action": "closed", "sender": {"login": "alice"}},  # no PR, no repo
        {
            "action": "closed",
            "pull_request": {"merged": True, "number": 7},
            "sender": {"login": "alice"},
        },  # merged, but no repository
        {
            "action": "closed",
            "pull_request": {"merged": True, "number": 7},
            "repository": {"full_name": "not-a-path"},
            "sender": {"login": "alice"},
        },  # merged, but malformed full_name
        {
            "action": "closed",
            "pull_request": {"number": 7},
            "repository": {"full_name": "o/r"},
            "sender": {"login": "alice"},
        },  # no merged flag — guard rejects before any provider work
    ],
)
async def test_handle_forgejo_merge_malformed_payload_is_noop(forgejo_auth, payload):
    """Broken payloads are a no-op: no provider creation, no learning, no raise."""
    from mira.platforms.forgejo import webhook as fw

    with (
        patch("mira.platforms.forgejo.webhook.create_provider") as cp,
        patch("mira.platforms.handlers.run_pr_merged_learning", new=AsyncMock()) as learn,
    ):
        await fw.handle_forgejo_merge(payload, forgejo_auth, BOT)
    cp.assert_not_called()
    learn.assert_not_awaited()
