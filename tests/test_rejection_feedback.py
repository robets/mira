"""Regression tests for rejection feedback learning (GitHub/GitLab/Forgejo).

Covers the trusted-thread-identity seam (``get_review_reply_context``), real
Forgejo API line extraction (``position``/``original_position``), the four
rejection write sites now recording parsed metadata from the *verified bot
original* (never the incoming human body or reply coordinates), dedupe of
webhook replays via ``list_feedback``, and that unverified roots produce no
signals and never reach the paid Q&A path.
"""

from __future__ import annotations

import re
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from mira.analysis import feedback as feedback_mod
from mira.config import CommandsConfig, MiraConfig
from mira.index.store import IndexStore
from mira.models import BotThreadRecord, PRInfo
from mira.platforms.forgejo.auth import ForgejoTokenAuth
from mira.platforms.gitlab.auth import GitLabTokenAuth
from mira.providers.forgejo import ForgejoProvider
from mira.providers.github import GitHubProvider, _async_constant
from mira.providers.gitlab import GitLabProvider

BOT = "mira-bot"

# Exactly what mira.providers.formatting.format_comment_body renders for a
# bug/blocker finding — the metadata the rejection events must carry.
BOT_BODY = "**Bug**  \n\U0001f6d1 Blocker — must fix before merge\n\n**Null deref**\n\nGuard the empty input.\n"


def _pr_info(platform: str = "github") -> PRInfo:
    return PRInfo(
        title="t",
        description="d",
        base_branch="main",
        head_branch="f",
        url="https://example.invalid/o/r/pulls/7",
        number=7,
        owner="o",
        repo="r",
        platform=platform,
        author="carol",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Forgejo provider: real source-shaped review comments
# (modules/structs/pull_review.go: position = new-side line,
#  original_position = old-side line, commit_id/original_commit_id,
#  pull_request_review_id, user, resolver)
# ─────────────────────────────────────────────────────────────────────────────


def _fj_comment(
    cid: int,
    *,
    review_id: int,
    username: str,
    body: str = "",
    path: str = "app.py",
    position: int = 0,
    original_position: int = 0,
    commit_id: str = "c1",
    original_commit_id: str = "c0",
    resolver: dict | None = None,
) -> dict[str, Any]:
    return {
        "id": cid,
        "body": body,
        "user": {"username": username},
        "resolver": resolver,
        "pull_request_review_id": review_id,
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z",
        "path": path,
        "commit_id": commit_id,
        "original_commit_id": original_commit_id,
        "diff_hunk": "@@ -1,3 +1,4 @@",
        "position": position,
        "original_position": original_position,
        "extra_lines_count": 0,
    }


def _fj_review(rid: int, username: str) -> dict[str, Any]:
    return {
        "id": rid,
        "user": {"username": username},
        "state": "COMMENT",
        "body": "",
        "commit_id": "c1",
        "stale": False,
    }


class _FJResp:
    def __init__(self, status: int = 200, json_data: Any = None):
        self.status_code = status
        self._json = json_data
        self.text = ""
        self.headers: dict[str, str] = {}

    def json(self) -> Any:
        return self._json

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("error", request=None, response=None)  # type: ignore[arg-type]


class _FJClient:
    def __init__(self, handler):  # noqa: ANN001
        self._handler = handler

    async def __aenter__(self):  # noqa: ANN204
        return self

    async def __aexit__(self, *args):  # noqa: ANN002, ANN204
        return False

    async def request(self, method: str, url: str, **kw):  # noqa: ANN001, ANN003, ANN201
        return self._handler(method, url, **kw)

    async def get(self, url: str, **kw):  # noqa: ANN001, ANN003, ANN201
        return self._handler("GET", url, **kw)


def _fj_api(reviews: list[dict], comments_by_review: dict[int, list[dict]]):  # noqa: ANN001, ANN201
    """Patch the Forgejo HTTP layer: /user identity, reviews, per-review comments."""

    def handler(method: str, url: str, **kw) -> _FJResp:
        if url.rstrip("/").endswith("/user"):
            return _FJResp(json_data={"username": BOT})
        m = re.search(r"/reviews/(\d+)/comments", url)
        if m:
            return _FJResp(json_data=comments_by_review.get(int(m.group(1)), []))
        if url.split("?")[0].endswith("/reviews"):
            return _FJResp(json_data=reviews)
        raise AssertionError(f"unexpected Forgejo URL: {url}")

    return patch("mira.providers.forgejo.httpx.AsyncClient", lambda *a, **k: _FJClient(handler))


class TestForgejoLineExtraction:
    """Forgejo's review-comment API serialises the file line as ``position``
    (new side) or ``original_position`` (old side) — never ``line`` or
    ``new_position``. Both merge-learning fetchers must use the real fields."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("user_field", ["login", "username"])
    async def test_pr_author_uses_user_object(self, user_field):
        provider = ForgejoProvider("tok")
        with patch.object(
            provider,
            "_request",
            new=AsyncMock(return_value=_FJResp(json_data={"user": {user_field: "carol"}})),
        ):
            pr = await provider.get_pr_info("https://forgejo.example/o/r/pulls/7")
        assert pr.author == "carol"

    @pytest.mark.asyncio
    async def test_get_all_bot_threads_uses_position_and_original_position(self):
        reviews = [_fj_review(5, BOT)]
        comments = {
            5: [
                _fj_comment(101, review_id=5, username=BOT, body=BOT_BODY, position=12),
                _fj_comment(102, review_id=5, username=BOT, body=BOT_BODY, original_position=7),
            ]
        }
        with _fj_api(reviews, comments):
            records = await ForgejoProvider("tok").get_all_bot_threads(_pr_info("forgejo"))
        assert [(r.path, r.line) for r in records] == [("app.py", 12), ("app.py", 7)]

    @pytest.mark.asyncio
    async def test_get_human_review_comments_uses_position_and_original_position(self):
        reviews = [_fj_review(6, "alice")]
        comments = {
            6: [
                _fj_comment(201, review_id=6, username="alice", body="use a constant", position=30),
                _fj_comment(
                    202, review_id=6, username="alice", body="dead code", original_position=4
                ),
            ]
        }
        with _fj_api(reviews, comments):
            out = await ForgejoProvider("tok").get_human_review_comments(_pr_info("forgejo"), BOT)
        assert [(c.path, c.line, c.author) for c in out] == [
            ("app.py", 30, "alice"),
            ("app.py", 4, "alice"),
        ]


class TestForgejoGetReviewReplyContext:
    def _provider(self) -> ForgejoProvider:
        return ForgejoProvider("tok")

    @pytest.mark.asyncio
    async def test_unique_bot_original_same_anchor(self):
        """Reply inherits the original's review/path/position/commits — the
        unique earlier bot comment at that anchor is the finding."""
        reviews = [_fj_review(5, BOT)]
        comments = {
            5: [
                _fj_comment(101, review_id=5, username=BOT, body=BOT_BODY, position=12),
                _fj_comment(
                    305, review_id=5, username="alice", body="@mira-bot reject", position=12
                ),
            ]
        }
        with _fj_api(reviews, comments):
            rec = await self._provider().get_review_reply_context(_pr_info("forgejo"), 305)
        assert rec is not None
        assert rec.thread_id == "101"
        assert rec.path == "app.py"
        assert rec.line == 12
        assert rec.body == BOT_BODY  # the trusted original, not the reply
        assert rec.is_resolved is False

    @pytest.mark.asyncio
    async def test_old_side_position_matches(self):
        """Both position sides: an old-side (original_position) reply resolves
        against an old-side bot original."""
        reviews = [_fj_review(5, BOT)]
        comments = {
            5: [
                _fj_comment(101, review_id=5, username=BOT, body=BOT_BODY, original_position=7),
                _fj_comment(305, review_id=5, username="alice", body="why?", original_position=7),
            ]
        }
        with _fj_api(reviews, comments):
            rec = await self._provider().get_review_reply_context(_pr_info("forgejo"), 305)
        assert rec is not None
        assert rec.line == 7

    @pytest.mark.asyncio
    async def test_wrong_side_does_not_match(self):
        """position=7 (new side) must not match a candidate at
        original_position=7 (old side) — different sides, no guessing."""
        reviews = [_fj_review(5, BOT)]
        comments = {
            5: [
                _fj_comment(101, review_id=5, username=BOT, body=BOT_BODY, position=7),
                _fj_comment(305, review_id=5, username="alice", body="why?", original_position=7),
            ]
        }
        with _fj_api(reviews, comments):
            rec = await self._provider().get_review_reply_context(_pr_info("forgejo"), 305)
        assert rec is None

    @pytest.mark.asyncio
    async def test_ambiguous_candidates_returns_none(self):
        reviews = [_fj_review(5, BOT)]
        comments = {
            5: [
                _fj_comment(101, review_id=5, username=BOT, body=BOT_BODY, position=12),
                _fj_comment(102, review_id=5, username=BOT, body=BOT_BODY, position=12),
                _fj_comment(305, review_id=5, username="alice", body="nope", position=12),
            ]
        }
        with _fj_api(reviews, comments):
            rec = await self._provider().get_review_reply_context(_pr_info("forgejo"), 305)
        assert rec is None

    @pytest.mark.asyncio
    async def test_human_only_original_returns_none(self):
        reviews = [_fj_review(5, "alice")]
        comments = {
            5: [
                _fj_comment(101, review_id=5, username="alice", body="human note", position=12),
                _fj_comment(305, review_id=5, username="bob", body="@mira-bot reject", position=12),
            ]
        }
        with _fj_api(reviews, comments):
            rec = await self._provider().get_review_reply_context(_pr_info("forgejo"), 305)
        assert rec is None

    @pytest.mark.asyncio
    async def test_incoming_id_not_a_review_comment_returns_none(self):
        reviews = [_fj_review(5, BOT)]
        comments = {5: [_fj_comment(101, review_id=5, username=BOT, body=BOT_BODY, position=12)]}
        with _fj_api(reviews, comments):
            rec = await self._provider().get_review_reply_context(_pr_info("forgejo"), 999)
        assert rec is None

    @pytest.mark.asyncio
    async def test_incoming_is_the_bot_original_itself_returns_none(self):
        """Not a reply: no *earlier* bot comment at the anchor → no guessing."""
        reviews = [_fj_review(5, BOT)]
        comments = {5: [_fj_comment(101, review_id=5, username=BOT, body=BOT_BODY, position=12)]}
        with _fj_api(reviews, comments):
            rec = await self._provider().get_review_reply_context(_pr_info("forgejo"), 101)
        assert rec is None

    @pytest.mark.asyncio
    async def test_different_commit_does_not_match(self):
        reviews = [_fj_review(5, BOT)]
        comments = {
            5: [
                _fj_comment(
                    101,
                    review_id=5,
                    username=BOT,
                    body=BOT_BODY,
                    position=12,
                    commit_id="other",
                ),
                _fj_comment(305, review_id=5, username="alice", body="nope", position=12),
            ]
        }
        with _fj_api(reviews, comments):
            rec = await self._provider().get_review_reply_context(_pr_info("forgejo"), 305)
        assert rec is None

    @pytest.mark.asyncio
    async def test_api_failure_degrades_to_none(self):
        def handler(method: str, url: str, **kw) -> _FJResp:
            if url.rstrip("/").endswith("/user"):
                return _FJResp(json_data={"username": BOT})
            return _FJResp(status=500, json_data=None)

        with patch("mira.providers.forgejo.httpx.AsyncClient", lambda *a, **k: _FJClient(handler)):
            rec = await ForgejoProvider("tok").get_review_reply_context(_pr_info("forgejo"), 305)
        assert rec is None


# ─────────────────────────────────────────────────────────────────────────────
# GitLab provider: scan discussions containing the incoming note id,
# verify the bot root (never the reply)
# ─────────────────────────────────────────────────────────────────────────────


def _gl_discussion(
    disc_id: str,
    root: dict[str, Any],
    replies: list[dict[str, Any]],
) -> dict[str, Any]:
    return {"id": disc_id, "notes": [root, *replies]}


def _gl_note(
    nid: int,
    *,
    username: str,
    body: str,
    type_: str = "Note",
    resolved: bool = False,
    position: dict | None = None,
    system: bool = False,
) -> dict[str, Any]:
    return {
        "id": nid,
        "type": type_,
        "body": body,
        "author": {"username": username},
        "resolved": resolved,
        "system": system,
        "position": position if position is not None else {"new_path": "app.py", "new_line": 12},
    }


def _gl_provider(discussions: list[dict]) -> GitLabProvider:
    prov = GitLabProvider("tok")
    prov._discussions = AsyncMock(return_value=discussions)  # type: ignore[method-assign]
    prov._self_username = AsyncMock(return_value=BOT)  # type: ignore[method-assign]
    return prov


class TestGitLabGetReviewReplyContext:
    @pytest.mark.asyncio
    async def test_returns_bot_root_not_reply(self):
        disc = _gl_discussion(
            "d1",
            _gl_note(
                100,
                username=BOT,
                body=BOT_BODY,
                type_="DiffNote",
                position={"new_path": "app.py", "new_line": 12},
            ),
            [_gl_note(305, username="alice", body="@mira-bot reject")],
        )
        prov = _gl_provider([disc])
        rec = await prov.get_review_reply_context(_pr_info("gitlab"), 305)
        assert rec is not None
        assert rec.thread_id == "d1"
        assert rec.body == BOT_BODY  # the original root, never the reply body
        assert rec.path == "app.py"
        assert rec.line == 12
        assert rec.is_resolved is False

    @pytest.mark.asyncio
    async def test_resolved_thread_still_returns_record(self):
        disc = _gl_discussion(
            "d1",
            _gl_note(100, username=BOT, body=BOT_BODY, type_="DiffNote", resolved=True),
            [_gl_note(305, username="alice", body="rejected earlier")],
        )
        rec = await _gl_provider([disc]).get_review_reply_context(_pr_info("gitlab"), 305)
        assert rec is not None
        assert rec.is_resolved is True

    @pytest.mark.asyncio
    async def test_human_root_returns_none(self):
        disc = _gl_discussion(
            "d1",
            _gl_note(100, username="dave", body="human root", type_="DiffNote"),
            [_gl_note(305, username="alice", body="@mira-bot reject")],
        )
        assert await _gl_provider([disc]).get_review_reply_context(_pr_info("gitlab"), 305) is None

    @pytest.mark.asyncio
    async def test_note_not_in_any_discussion_returns_none(self):
        disc = _gl_discussion(
            "d1", _gl_note(100, username=BOT, body=BOT_BODY, type_="DiffNote"), []
        )
        assert await _gl_provider([disc]).get_review_reply_context(_pr_info("gitlab"), 999) is None

    @pytest.mark.asyncio
    async def test_unverifiable_bot_identity_returns_none(self):
        """If the bot's own identity can't be resolved, refuse to trust a root."""
        disc = _gl_discussion(
            "d1",
            _gl_note(100, username="whoever", body=BOT_BODY, type_="DiffNote"),
            [_gl_note(305, username="alice", body="reject")],
        )
        prov = GitLabProvider("tok")
        prov._discussions = AsyncMock(return_value=[disc])  # type: ignore[method-assign]
        prov._self_username = AsyncMock(return_value="")  # type: ignore[method-assign]
        assert await prov.get_review_reply_context(_pr_info("gitlab"), 305) is None


# ─────────────────────────────────────────────────────────────────────────────
# GitHub provider: augmented comment-thread query (root fields, resolved)
# ─────────────────────────────────────────────────────────────────────────────


class TestGitHubGetReviewReplyContext:
    def _make_provider(self) -> GitHubProvider:
        provider = GitHubProvider.__new__(GitHubProvider)
        provider._token_supplier = _async_constant("test-token")
        return provider

    def _resp(self, threads: list[dict], viewer: str = BOT, has_next: bool = False) -> dict:
        return {
            "data": {
                "viewer": {"login": viewer},
                "repository": {
                    "pullRequest": {
                        "reviewThreads": {
                            "pageInfo": {"hasNextPage": has_next, "endCursor": None},
                            "nodes": threads,
                        }
                    }
                },
            }
        }

    def _thread(
        self,
        tid: str,
        *,
        resolved: bool,
        root_author: str = BOT,
        body: str = BOT_BODY,
        path: str = "app.py",
        line: int = 12,
        root_id: str = "MDI0Ol_root",
        reply_id: str = "MDI0Ol_reply",
        reply_db_id: int = 305,
    ) -> dict:
        return {
            "id": tid,
            "isResolved": resolved,
            "isOutdated": False,
            "comments": {
                "nodes": [
                    {
                        "id": root_id,
                        "databaseId": 101,
                        "author": {"login": root_author},
                        "body": body,
                        "path": path,
                        "line": line,
                        "originalLine": None,
                    },
                    {
                        "id": reply_id,
                        "databaseId": reply_db_id,
                        "author": {"login": "alice"},
                        "body": "@mira-bot reject",
                        "path": path,
                        "line": line,
                        "originalLine": None,
                    },
                ]
            },
        }

    async def _run(self, resp: dict, incoming: int) -> BotThreadRecord | None:
        async def _mock_post(self, url, **kwargs):  # noqa: ANN001, ANN003
            return httpx.Response(200, json=resp, request=httpx.Request("POST", url))

        with patch.object(httpx.AsyncClient, "post", _mock_post):
            return await self._make_provider().get_review_reply_context(_pr_info(), incoming)

    @pytest.mark.asyncio
    async def test_matches_by_database_id_and_returns_root_fields(self):
        rec = await self._run(self._resp([self._thread("PRRT_1", resolved=False)]), 305)
        assert rec is not None
        assert rec.thread_id == "PRRT_1"
        assert rec.body == BOT_BODY
        assert rec.path == "app.py"
        assert rec.line == 12
        assert rec.is_resolved is False

    @pytest.mark.asyncio
    async def test_resolved_thread_still_returns_record(self):
        """GH already-resolved learning: a reject arriving after manual
        resolution must still yield the trusted record."""
        rec = await self._run(self._resp([self._thread("PRRT_1", resolved=True)]), 305)
        assert rec is not None
        assert rec.is_resolved is True
        assert rec.body == BOT_BODY

    @pytest.mark.asyncio
    async def test_human_root_returns_none(self):
        rec = await self._run(
            self._resp([self._thread("PRRT_1", resolved=False, root_author="dave")]), 305
        )
        assert rec is None

    @pytest.mark.asyncio
    async def test_missing_bot_and_root_identity_returns_none(self):
        thread = self._thread("PRRT_1", resolved=False, root_author="")
        thread["comments"]["nodes"][0]["author"] = None
        assert await self._run(self._resp([thread], viewer=""), 305) is None

    @pytest.mark.asyncio
    async def test_comment_not_in_any_thread_returns_none(self):
        rec = await self._run(self._resp([self._thread("PRRT_1", resolved=False)]), 999)
        assert rec is None

    @pytest.mark.asyncio
    async def test_get_thread_id_for_comment_behavior_preserved(self):
        """The augmented query must not break the existing thread-id lookup:
        unresolved → id, already resolved → None."""
        unresolved_resp = self._resp([self._thread("PRRT_1", resolved=False)])
        resolved_resp = self._resp([self._thread("PRRT_1", resolved=True)])

        def _post_for(resp: dict):  # noqa: ANN202
            async def _mock_post(self, url, **kwargs):  # noqa: ANN001, ANN003
                return httpx.Response(200, json=resp, request=httpx.Request("POST", url))

            return _mock_post

        with patch.object(httpx.AsyncClient, "post", _post_for(unresolved_resp)):
            tid = await self._make_provider().get_thread_id_for_comment("MDI0Ol_reply", _pr_info())
        assert tid == "PRRT_1"

        with patch.object(httpx.AsyncClient, "post", _post_for(resolved_resp)):
            tid = await self._make_provider().get_thread_id_for_comment("MDI0Ol_reply", _pr_info())
        assert tid is None


# ─────────────────────────────────────────────────────────────────────────────
# Shared helper: metadata-parsed, deduped rejection feedback
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def store(tmp_path):  # noqa: ANN201
    s = IndexStore(str(tmp_path / "fb.db"))
    yield s
    s.close()


class TestRecordRejectedFeedback:
    def _call(self, store: IndexStore, **overrides):  # noqa: ANN001, ANN201
        from mira.platforms.handlers import _record_rejected_feedback

        kwargs: dict[str, Any] = {
            "pr_number": 7,
            "pr_url": "https://example.invalid/o/r/pull/7",
            "original_body": BOT_BODY,
            "comment_path": "src/a.py",
            "comment_line": 12,
            "actor": "alice",
            "pr_author": "carol",
        }
        kwargs.update(overrides)
        return _record_rejected_feedback(store, **kwargs)

    def test_records_parsed_metadata_from_original(self, store):
        assert self._call(store) is True
        (ev,) = store.list_feedback()
        assert ev.signal == "rejected"
        assert ev.comment_category == "bug"
        assert ev.comment_severity == "blocker"
        assert ev.comment_title == "Null deref"
        assert ev.comment_path == "src/a.py"
        assert ev.comment_line == 12
        assert ev.actor == "alice"
        assert ev.pr_author == "carol"

    def test_unparseable_original_records_nothing(self, store):
        assert self._call(store, original_body="just prose, no structure") is False
        assert store.list_feedback() == []

    def test_empty_original_records_nothing(self, store):
        assert self._call(store, original_body="") is False
        assert store.list_feedback() == []

    def test_duplicate_actor_finding_is_deduped(self, store):
        assert self._call(store) is True
        # webhook replay of the same reject by the same actor/finding
        assert self._call(store) is False
        assert len(store.list_feedback()) == 1

    def test_different_actor_still_records(self, store):
        assert self._call(store) is True
        assert self._call(store, actor="bob") is True
        assert len(store.list_feedback()) == 2

    def test_different_finding_at_same_location_still_records(self, store):
        assert self._call(store) is True
        different_body = BOT_BODY.replace("Null deref", "Wrong result")
        assert self._call(store, original_body=different_body) is True
        assert len(store.list_feedback()) == 2

    def test_legacy_uncategorized_reject_does_not_block_metadata(self, store):
        store.record_feedback(
            pr_number=7,
            pr_url="https://example.invalid/o/r/pull/7",
            comment_path="src/a.py",
            comment_line=12,
            comment_category="",
            comment_severity="",
            comment_title="",
            signal="rejected",
            actor="alice",
        )
        assert self._call(store) is True
        assert any(e.comment_category == "bug" for e in store.list_feedback())

    def test_metadata_feeds_synthesize_rules_to_pending_rule(self, store, monkeypatch):
        """The whole point of parsed metadata: at threshold, synthesize_rules
        must create a *pending* reject_pattern rule."""
        monkeypatch.setattr(feedback_mod, "_MIN_REJECTS_PER_DIR", 2)
        assert self._call(store, comment_path="src/a.py", comment_line=12) is True
        assert self._call(store, comment_path="src/b.py", comment_line=4) is True
        created = feedback_mod.synthesize_rules(store)
        assert created >= 1
        rules = [r for r in store.list_learned_rules() if r.source_signal == "reject_pattern"]
        assert any(
            r.category == "bug" and r.path_pattern == "src/**" and r.status == "pending"
            for r in rules
        )


# ─────────────────────────────────────────────────────────────────────────────
# GitHub handler: reject learns from the verified context (root path/line +
# metadata), records even when resolved / resolve fails, none when untrusted
# ─────────────────────────────────────────────────────────────────────────────


def _gh_reject_payload(body: str = "@mira-bot reject") -> dict[str, Any]:
    return {
        "installation": {"id": 1},
        "action": "created",
        "comment": {
            "id": 305,
            "body": body,
            "node_id": "MDI0Ol_reply",
            "user": {"login": "alice", "id": 7},
        },
        "pull_request": {"number": 42, "user": {"login": "carol"}},
        "repository": {"owner": {"login": "testowner"}, "name": "testrepo"},
    }


class _StoreProxy:
    def __init__(self, inner: IndexStore) -> None:
        self._inner = inner

    def __getattr__(self, name):  # noqa: ANN001, ANN204  # type: ignore[no-untyped-def]
        return getattr(self._inner, name)

    def close(self) -> None:
        pass


def _gh_provider() -> AsyncMock:
    provider = AsyncMock()
    provider.get_repo_permission = AsyncMock(return_value="write")
    provider.get_thread_id_for_comment = AsyncMock(return_value="PRRT_1")
    provider.resolve_threads = AsyncMock(return_value=1)
    provider.post_comment = AsyncMock()
    provider.get_review_reply_context = AsyncMock(
        return_value=BotThreadRecord(
            thread_id="PRRT_1",
            path="src/a.py",
            line=12,
            body=BOT_BODY,
            is_resolved=False,
        )
    )
    return provider


async def _run_gh_reject(
    store: IndexStore, provider: AsyncMock, payload: dict | None = None
) -> None:
    from mira.platforms.github import webhook as ghw

    with (
        patch("mira.platforms.github.webhook.create_provider", return_value=provider),
        patch(
            "mira.platforms.github.webhook._open_store",
            lambda owner, repo, platform="github": _StoreProxy(store),
        ),
    ):
        await ghw.handle_thread_reject(payload or _gh_reject_payload(), AsyncMock(), "mira-bot")


class TestGitHubRejectLearning:
    @pytest.mark.asyncio
    async def test_records_verified_root_metadata(self, store):
        await _run_gh_reject(store, _gh_provider())
        (ev,) = store.list_feedback()
        assert ev.comment_category == "bug"
        assert ev.comment_severity == "blocker"
        assert ev.comment_title == "Null deref"
        assert ev.comment_path == "src/a.py"  # root coords, not reply coords
        assert ev.comment_line == 12
        assert ev.actor == "alice"
        assert ev.pr_author == "carol"

    @pytest.mark.asyncio
    async def test_already_resolved_thread_still_records_intent(self, store):
        provider = _gh_provider()
        provider.get_thread_id_for_comment = AsyncMock(return_value=None)  # already resolved
        provider.get_review_reply_context = AsyncMock(
            return_value=BotThreadRecord(
                thread_id="PRRT_1", path="src/a.py", line=12, body=BOT_BODY, is_resolved=True
            )
        )
        await _run_gh_reject(store, provider)
        provider.resolve_threads.assert_not_awaited()
        assert len(store.list_feedback()) == 1

    @pytest.mark.asyncio
    async def test_resolve_failure_still_records_and_posts_apology(self, store):
        provider = _gh_provider()
        provider.resolve_threads = AsyncMock(side_effect=RuntimeError("network down"))
        await _run_gh_reject(store, provider)
        provider.post_comment.assert_awaited_once()
        assert "couldn't dismiss" in provider.post_comment.call_args[0][1]
        assert len(store.list_feedback()) == 1

    @pytest.mark.asyncio
    async def test_unverified_context_records_no_bogus_signal(self, store):
        provider = _gh_provider()
        provider.get_review_reply_context = AsyncMock(return_value=None)
        await _run_gh_reject(store, provider)
        provider.resolve_threads.assert_awaited_once()  # UX preserved for untrusted threads
        assert store.list_feedback() == []

    @pytest.mark.asyncio
    async def test_webhook_replay_is_stable(self, store):
        provider = _gh_provider()
        await _run_gh_reject(store, provider)
        await _run_gh_reject(store, provider)
        assert len(store.list_feedback()) == 1


# ─────────────────────────────────────────────────────────────────────────────
# Forgejo note handler: real context resolution replaces the synthetic
# payload-path gate
# ─────────────────────────────────────────────────────────────────────────────


def _fj_note_payload(note: str) -> dict[str, Any]:
    """Source-shaped Forgejo ``issue_comment`` webhook for an inline reply:
    CreateIssueComment emits api.Comment — id/body/user ONLY. No path, no
    line, no parent, despite being a reply to a review comment."""
    return {
        "action": "created",
        "is_pull": True,
        "comment": {"id": 305, "body": note, "user": {"username": "alice", "id": 7}},
        "issue": {"number": 7, "user": {"username": "carol"}},
        "pull_request": {"number": 7},
        "repository": {"full_name": "o/r", "html_url": "https://fj.example/o/r"},
        "sender": {"login": "alice"},
    }


def _fj_provider(context: BotThreadRecord | None) -> AsyncMock:
    provider = AsyncMock()
    provider.get_repo_permission = AsyncMock(return_value="write")
    provider.get_pr_info = AsyncMock(
        return_value=PRInfo(
            title="t",
            description="d",
            base_branch="main",
            head_branch="f",
            url="https://fj.example/o/r/pulls/7",
            number=7,
            owner="o",
            repo="r",
            platform="forgejo",
            author="carol",
        )
    )
    provider.get_review_reply_context = AsyncMock(return_value=context)
    return provider


def _finding_context() -> BotThreadRecord:
    return BotThreadRecord(
        thread_id="5", path="src/a.py", line=12, body=BOT_BODY, is_resolved=False
    )


async def _run_fj_note(
    store: IndexStore,
    note: str,
    context: BotThreadRecord | None,
    config: MiraConfig | None = None,
) -> tuple[AsyncMock, AsyncMock, AsyncMock]:
    from mira.platforms.forgejo import webhook as fjwt

    auth = ForgejoTokenAuth("tok")
    auth.get_bot_identity = AsyncMock(return_value=BOT)  # type: ignore[method-assign]
    provider = _fj_provider(context)
    with (
        patch("mira.platforms.forgejo.webhook.create_provider", return_value=provider),
        patch("mira.platforms.forgejo.webhook.load_config", return_value=config or MiraConfig()),
        patch("mira.platforms.handlers._open_store", lambda o, r, p="github": _StoreProxy(store)),
        patch("mira.platforms.handlers.run_pr_command", new=AsyncMock()) as rpc,
        patch("mira.platforms.handlers.run_thread_reply", new=AsyncMock()) as rtr,
    ):
        await fjwt.handle_forgejo_note(_fj_note_payload(note), auth, BOT)
    return provider, rpc, rtr


class TestForgejoNoteHandler:
    @pytest.mark.asyncio
    async def test_source_shaped_pathless_reply_resolved_via_api_context(self, store):
        """The payload carries no path — identity comes from the API context,
        and the reject records real metadata instead of falling to Q&A."""
        provider, rpc, rtr = await _run_fj_note(store, "@mira-bot reject", _finding_context())
        provider.get_review_reply_context.assert_awaited_once()
        rpc.assert_not_awaited()
        rtr.assert_not_awaited()
        (ev,) = store.list_feedback()
        assert ev.comment_category == "bug"
        assert ev.comment_title == "Null deref"
        assert ev.comment_path == "src/a.py"  # from the context root
        assert ev.comment_line == 12
        assert ev.actor == "alice"
        assert ev.pr_author == "carol"

    @pytest.mark.asyncio
    async def test_unresolved_reject_never_paid_qa_nor_bogus_feedback(self, store):
        provider, rpc, rtr = await _run_fj_note(store, "@mira-bot reject", None)
        rpc.assert_not_awaited()  # never the paid free-form Q&A path
        rtr.assert_not_awaited()  # never the classifier path
        assert store.list_feedback() == []  # never a bogus empty-metadata event

    @pytest.mark.asyncio
    async def test_unauthorized_reject_denied(self, store):
        config = MiraConfig(
            commands=CommandsConfig.model_validate({"reject": {"blocked_users": ["alice"]}})
        )
        _, rpc, rtr = await _run_fj_note(
            store, "@mira-bot reject", _finding_context(), config=config
        )
        assert store.list_feedback() == []

    @pytest.mark.asyncio
    async def test_freeform_gets_trusted_original_not_human_body(self, store):
        provider, rpc, rtr = await _run_fj_note(
            store, "@mira-bot are you sure?", _finding_context()
        )
        rtr.assert_awaited_once()
        rpc.assert_not_awaited()
        kwargs = rtr.await_args.kwargs
        assert kwargs["original_suggestion"] == BOT_BODY  # trusted finding body ...
        assert "are you sure?" not in kwargs["original_suggestion"]  # ... never the reply
        assert kwargs["comment_path"] == "src/a.py"
        assert kwargs["comment_line"] == 12
        assert kwargs["thread_id"] == "5"

    @pytest.mark.asyncio
    async def test_unverified_freeform_falls_to_general_commands(self, store):
        """Non-inline / unresolvable comments keep the general command UX."""
        _, rpc, rtr = await _run_fj_note(store, "@mira-bot are you sure?", None)
        rtr.assert_not_awaited()
        rpc.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_unparseable_original_records_nothing(self, store):
        ctx = BotThreadRecord(thread_id="5", path="src/a.py", line=12, body="hi", is_resolved=False)
        await _run_fj_note(store, "@mira-bot reject", ctx)
        assert store.list_feedback() == []

    @pytest.mark.asyncio
    async def test_replay_stable(self, store):
        await _run_fj_note(store, "@mira-bot reject", _finding_context())
        await _run_fj_note(store, "@mira-bot reject", _finding_context())
        assert len(store.list_feedback()) == 1

    @pytest.mark.asyncio
    async def test_api_reply_to_pending_learning_survives_merge(self, store, monkeypatch):
        from mira.platforms import handlers
        from mira.platforms.forgejo import webhook

        provider = ForgejoProvider("tok")
        originals = [
            _fj_comment(i, review_id=5, username=BOT, body=BOT_BODY, path="src/a.py", position=line)
            for i, line in zip(range(101, 105), [12, 20, 30, 40], strict=True)
        ]
        replies = [
            _fj_comment(
                i,
                review_id=5,
                username="alice",
                body="@mira-bot reject",
                path="src/a.py",
                position=line,
            )
            for i, line in zip(range(305, 308), [12, 20, 30], strict=True)
        ]

        def request(method, url, **kwargs):
            url = url.split("?", 1)[0]
            if url.endswith("/user"):
                return _FJResp(json_data={"username": BOT})
            if url.endswith("/pulls/7"):
                return _FJResp(json_data={"number": 7, "user": {"login": "carol"}})
            if url.endswith("/reviews"):
                return _FJResp(json_data=[_fj_review(5, BOT)])
            if url.endswith("/reviews/5/comments"):
                return _FJResp(json_data=originals + replies)
            raise AssertionError(f"Unexpected API request: {method} {url}")

        monkeypatch.setattr(
            "mira.providers.forgejo.httpx.AsyncClient", lambda *args, **kwargs: _FJClient(request)
        )
        monkeypatch.setattr(webhook, "create_provider", lambda *args: provider)
        config = MiraConfig(
            commands=CommandsConfig(allowed_users=["alice"], allow_repo_write_access=False)
        )
        monkeypatch.setattr(webhook, "load_config", lambda: config)
        monkeypatch.setattr(handlers, "_open_store", lambda *args: _StoreProxy(store))
        monkeypatch.setattr(feedback_mod, "_MIN_REJECTS_PER_DIR", 3)
        auth = ForgejoTokenAuth("tok")
        auth.get_bot_identity = AsyncMock(return_value=BOT)
        for reply in replies:
            payload = _fj_note_payload(reply["body"])
            payload["comment"]["id"] = reply["id"]
            await webhook.handle_forgejo_note(payload, auth, BOT)
            await webhook.handle_forgejo_note(payload, auth, BOT)

        assert len(store.list_feedback()) == 3
        pr = await provider.get_pr_info("https://fj.example/o/r/pulls/7")
        await handlers.run_pr_merged_learning(provider, pr, BOT, "carol", platform="forgejo")
        events = store.list_feedback()
        assert [(e.comment_line, e.comment_category) for e in events if e.signal == "accepted"] == [
            (40, "bug")
        ]
        assert len([e for e in events if e.signal == "rejected"]) == 3
        rules = store.list_learned_rules()
        assert len(rules) == 1
        assert rules[0].status == "pending"
        assert rules[0].source_signal == "reject_pattern"
        assert rules[0].sample_count == 3


# ─────────────────────────────────────────────────────────────────────────────
# GitLab note handler: verified bot root for reject + free-form
# ─────────────────────────────────────────────────────────────────────────────


def _gl_note_payload(note: str) -> dict[str, Any]:
    return {
        "object_attributes": {
            "note": note,
            "noteable_type": "MergeRequest",
            "id": 305,
            "discussion_id": "d1",
            "position": {"new_path": "src/a.py", "new_line": 12},
        },
        "merge_request": {"iid": 7, "url": "https://gitlab.example/g/p/-/merge_requests/7"},
        "project": {"path_with_namespace": "g/p", "web_url": "https://gitlab.example/g/p"},
        "user": {"username": "alice", "id": 7},
    }


def _gl_provider_mock(context: BotThreadRecord | None) -> AsyncMock:
    provider = AsyncMock()
    provider.get_repo_permission = AsyncMock(return_value="write")
    provider.get_pr_info = AsyncMock(
        return_value=PRInfo(
            title="t",
            description="d",
            base_branch="main",
            head_branch="f",
            url="https://gitlab.example/g/p/-/merge_requests/7",
            number=7,
            owner="g",
            repo="p",
            platform="gitlab",
            author="carol",
        )
    )
    provider.resolve_threads = AsyncMock(return_value=1)
    provider.get_review_reply_context = AsyncMock(return_value=context)
    return provider


async def _run_gl_note(
    store: IndexStore, note: str, context: BotThreadRecord | None
) -> tuple[AsyncMock, AsyncMock, AsyncMock]:
    from mira.platforms.gitlab import webhook as gwt

    auth = GitLabTokenAuth("tok")
    auth.get_bot_identity = AsyncMock(return_value=BOT)  # type: ignore[method-assign]
    provider = _gl_provider_mock(context)
    with (
        patch("mira.platforms.gitlab.webhook.create_provider", return_value=provider),
        patch("mira.platforms.gitlab.webhook.load_config", return_value=MiraConfig()),
        patch("mira.platforms.handlers._open_store", lambda o, r, p="github": _StoreProxy(store)),
        patch("mira.platforms.handlers.run_pr_command", new=AsyncMock()) as rpc,
        patch("mira.platforms.handlers.run_thread_reply", new=AsyncMock()) as rtr,
    ):
        await gwt.handle_gitlab_note(_gl_note_payload(note), auth, BOT)
    return provider, rpc, rtr


class TestGitLabNoteHandler:
    @pytest.mark.asyncio
    async def test_reject_resolves_and_records_verified_metadata(self, store):
        provider, rpc, rtr = await _run_gl_note(store, "@mira-bot reject", _finding_context())
        provider.resolve_threads.assert_awaited_once()
        assert provider.resolve_threads.await_args[0][1] == ["5"]
        rpc.assert_not_awaited()
        rtr.assert_not_awaited()
        (ev,) = store.list_feedback()
        assert ev.comment_category == "bug"
        assert ev.comment_line == 12
        assert ev.pr_author == "carol"

    @pytest.mark.asyncio
    async def test_unverified_root_no_resolve_no_feedback_no_qa(self, store):
        provider, rpc, rtr = await _run_gl_note(store, "@mira-bot reject", None)
        provider.resolve_threads.assert_not_awaited()
        assert store.list_feedback() == []
        rpc.assert_not_awaited()
        rtr.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_unauthorized_reject_denied(self, store):
        from mira.platforms.gitlab import webhook as gwt

        auth = GitLabTokenAuth("tok")
        auth.get_bot_identity = AsyncMock(return_value=BOT)  # type: ignore[method-assign]
        provider = _gl_provider_mock(_finding_context())
        config = MiraConfig(
            commands=CommandsConfig.model_validate({"reject": {"blocked_users": ["alice"]}})
        )
        with (
            patch("mira.platforms.gitlab.webhook.create_provider", return_value=provider),
            patch("mira.platforms.gitlab.webhook.load_config", return_value=config),
            patch("mira.platforms.handlers.run_pr_command", new=AsyncMock()) as rpc,
        ):
            await gwt.handle_gitlab_note(_gl_note_payload("@mira-bot reject"), auth, BOT)
        provider.resolve_threads.assert_not_awaited()
        rpc.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_freeform_uses_verified_root_body(self, store):
        _, rpc, rtr = await _run_gl_note(store, "@mira-bot why?", _finding_context())
        rtr.assert_awaited_once()
        assert rtr.await_args.kwargs["original_suggestion"] == BOT_BODY
        assert rtr.await_args.kwargs["thread_id"] == "5"

    @pytest.mark.asyncio
    async def test_freeform_human_root_gets_empty_original_never_body(self, store):
        """Free-form on an unverified (human-root) thread still replies, but
        the classifier must not receive any untrusted/original body."""
        _, _, rtr = await _run_gl_note(store, "@mira-bot why?", None)
        rtr.assert_awaited_once()
        assert rtr.await_args.kwargs["original_suggestion"] == ""


# ─────────────────────────────────────────────────────────────────────────────
# run_thread_reply: disagreement learns with parsed metadata + dedupe,
# unauthorized reject stays denied, untrusted original records nothing
# ─────────────────────────────────────────────────────────────────────────────


async def _run_disagreement(store: IndexStore, original: str = BOT_BODY):
    from mira.platforms.handlers import run_thread_reply

    provider = MagicMock()
    provider.get_repo_permission = AsyncMock(return_value="write")
    provider.reply_to_review_comment = AsyncMock()
    provider.resolve_threads = AsyncMock()
    llm = MagicMock()
    llm.complete_with_tools = AsyncMock(
        return_value='{"intent": "disagreement", "reply": "Understood."}'
    )
    pr_info = _pr_info()
    with (
        patch("mira.platforms.handlers.load_config", return_value=MiraConfig()),
        patch("mira.platforms.handlers.create_llm", return_value=llm),
        patch("mira.platforms.handlers.llm_config_for"),
        patch("mira.platforms.handlers._open_store", lambda o, r, p="github": _StoreProxy(store)),
    ):
        await run_thread_reply(
            provider,
            pr_info,
            "I disagree",
            1,
            original_suggestion=original,
            thread_id="T1",
            comment_path="src/a.py",
            comment_line=12,
            actor="alice",
        )
    return provider, llm


class TestThreadReplyDisagreementLearning:
    @pytest.mark.asyncio
    async def test_records_parsed_metadata_and_pr_author(self, store):
        provider, _llm = await _run_disagreement(store)
        provider.resolve_threads.assert_awaited_once()
        (ev,) = store.list_feedback()
        assert ev.signal == "rejected"
        assert ev.comment_category == "bug"
        assert ev.comment_title == "Null deref"
        assert ev.comment_path == "src/a.py"
        assert ev.pr_author == "carol"

    @pytest.mark.asyncio
    async def test_duplicate_disagreement_replay_is_stable(self, store):
        await _run_disagreement(store)
        await _run_disagreement(store)
        assert len(store.list_feedback()) == 1

    @pytest.mark.asyncio
    async def test_no_metadata_no_event(self, store):
        """An untrusted/unparseable original must never synthesize an empty
        metadata event."""
        await _run_disagreement(store, original="")
        assert store.list_feedback() == []

    @pytest.mark.asyncio
    async def test_freeform_disagreement_denies_unauthorized_reject(self, store):
        from mira.platforms.handlers import run_thread_reply

        provider = MagicMock()
        provider.get_repo_permission = AsyncMock(return_value="write")
        provider.reply_to_review_comment = AsyncMock()
        provider.resolve_threads = AsyncMock()
        llm = MagicMock()
        llm.complete_with_tools = AsyncMock(
            return_value='{"intent": "disagreement", "reply": "Understood."}'
        )
        config = MiraConfig(
            commands=CommandsConfig.model_validate({"reject": {"blocked_users": ["alice"]}})
        )
        with (
            patch("mira.platforms.handlers.load_config", return_value=config),
            patch("mira.platforms.handlers.create_llm", return_value=llm),
            patch("mira.platforms.handlers.llm_config_for"),
            patch(
                "mira.platforms.handlers._open_store", lambda o, r, p="github": _StoreProxy(store)
            ),
        ):
            await run_thread_reply(
                provider,
                _pr_info(),
                "I disagree",
                1,
                original_suggestion=BOT_BODY,
                thread_id="T1",
                comment_path="src/a.py",
                comment_line=12,
                actor="alice",
            )
        provider.reply_to_review_comment.assert_awaited_once()  # ask allowed
        provider.resolve_threads.assert_not_awaited()  # reject denied
        assert store.list_feedback() == []
