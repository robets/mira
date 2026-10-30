"""Regression tests for Forgejo per-finding resolution replies.

Covers the review-comments transport Mira uses to record resolutions on
Forgejo (which has no native resolve endpoint in v16/v17):

* POST to the *original review's* comments URL with ``body, path,
  new_position, old_position, extra_lines_count`` copied from the comment
  GET's ``position``/``original_position``/``extra_lines_count`` (side and
  range preserved, stale/deleted anchors included).
* ``thread_id`` is the original comment ID, not the review ID.
* ``is_resolved`` comes from the native ``resolver`` field or a trusted
  bot-authored ``<!-- mira:resolved original=N sha=S -->`` marker at the
  finding's exact anchor; human-spoofed, mis-anchored, earlier or
  wrong-target markers resolve nothing.
* Dedup: one acknowledgment per original across reruns and pushes; complete
  (paginated) reads required before writes; read-back before counting
  success; per-ID failures isolated; no retries; no top-level fallback;
  writes refuse to run without a verified bot identity or head SHA.

All HTTP is mocked at the complete httpx transport boundary. The fake server
only answers routes shaped like the real Forgejo API; anything else returns
status 599 so an unexpected request fails the test loudly. The fake token is
``fake-token``; no network is ever reachable (``.invalid`` TLD).
"""

from __future__ import annotations

import base64
import json
import re
from urllib.parse import unquote

import httpx
import pytest

from mira.core.threads import resolve_verified_threads
from mira.exceptions import ProviderError
from mira.models import PRInfo
from mira.platforms import profiles
from mira.providers.forgejo import ForgejoProvider, _resolution_marker

API = "https://forge.invalid/api/v1"
BASE = "https://forge.invalid"
HEAD_SHA = "ab" * 20
OLD_SHA = "cd" * 20
BOT = "mira-bot"

_REVIEWS_RE = re.compile(r"^/api/v1/repos/o/r/pulls/1/reviews$")
_REVIEW_COMMENTS_RE = re.compile(r"^/api/v1/repos/o/r/pulls/1/reviews/(\d+)/comments$")
_REVIEW_COMMENT_RE = re.compile(r"^/api/v1/repos/o/r/pulls/1/reviews/(\d+)/comments/(\d+)$")


class FakeForgejo:
    """Complete HTTP boundary mock shaped like Forgejo's /api/v1."""

    def __init__(self) -> None:
        self.username: str | None = BOT
        self.user_status = 200
        self.user_payload: dict | None = None  # full /user response override
        self.reviews: list[list[dict]] = []  # pages of review JSON
        self.review_comments: dict[int, list[list[dict]]] = {}  # review id -> pages
        self.contents: dict[str, str] = {}  # path -> file text
        self.fail_comment_fetch: set[int] = set()  # review ids whose GET 500s
        self.fail_post: set[int] = set()  # review ids whose POST 500s
        self.fail_readback: set[int] = set()  # created comment ids whose GET 500s
        self.readback_override: dict[int, dict] = {}  # tamper a read-back payload
        self.nonlist_reviews = False  # reviews GET returns a JSON object
        self.requests: list[tuple[str, str, dict]] = []  # (method, url, headers)
        self.posts: list[tuple[str, dict]] = []  # (url, json payload) of *created* comments
        self.post_attempts: list[str] = []  # every POST attempt URL, success or not
        self._next_comment_id = 9000

    # ── transport ───────────────────────────────────────────────────

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append((request.method, str(request.url), dict(request.headers)))
        path = request.url.path

        if path == "/api/v1/user":
            if self.user_status != 200:
                return self._resp(self.user_status, {"message": "boom"})
            payload = (
                self.user_payload if self.user_payload is not None else {"username": self.username}
            )
            return self._resp(200, payload)

        if _REVIEWS_RE.match(path) and request.method == "GET":
            if self.nonlist_reviews:
                return self._resp(200, {"message": "not a list"})
            return self._paged(self.reviews, request)

        match = _REVIEW_COMMENTS_RE.match(path)
        if match:
            review_id = int(match.group(1))
            if request.method == "GET":
                if review_id in self.fail_comment_fetch:
                    return self._resp(500, {"message": "boom"})
                return self._paged(self.review_comments.get(review_id, [[]]), request)
            return self._create_comment(review_id, request)

        match = _REVIEW_COMMENT_RE.match(path)
        if match and request.method == "GET":
            review_id, comment_id = (int(g) for g in match.groups())
            if comment_id in self.fail_readback:
                return self._resp(500, {"message": "boom"})
            for page in self.review_comments.get(review_id, []):
                for comment in page:
                    if comment["id"] == comment_id:
                        return self._resp(200, self.readback_override.get(comment_id, comment))
            return self._resp(404, {"message": "not found"})

        if path.startswith("/api/v1/repos/o/r/contents/") and request.method == "GET":
            file_path = unquote(path[len("/api/v1/repos/o/r/contents/") :])
            if file_path not in self.contents:
                return self._resp(404, {"message": "not found"})
            encoded = base64.b64encode(self.contents[file_path].encode()).decode()
            return self._resp(200, {"content": encoded})

        if path == "/api/v1/repos/o/r/issues/1/comments" and request.method == "GET":
            return self._paged([[]], request)

        return self._resp(599, {"message": f"unexpected {request.method} {path}"})

    def _resp(self, status: int, payload) -> httpx.Response:
        return httpx.Response(status, json=payload)

    def _paged(self, pages: list[list[dict]], request: httpx.Request) -> httpx.Response:
        page = int(request.url.params.get("page", 1))
        if page > len(pages):
            return self._resp(200, [])
        headers = {}
        if page < len(pages):
            base = str(request.url).split("?", 1)[0]
            headers["link"] = f'<{base}?limit=100&page={page + 1}>; rel="next"'
        return httpx.Response(200, json=pages[page - 1], headers=headers)

    def _create_comment(self, review_id: int, request: httpx.Request) -> httpx.Response:
        self.post_attempts.append(str(request.url))
        if review_id in self.fail_post:
            return self._resp(500, {"message": "boom"})
        payload = json.loads(request.content)
        originals = [
            comment
            for page in self.review_comments.get(review_id, [])
            for comment in page
            if comment["path"] == payload.get("path")
            and (comment.get("position") or 0) == payload.get("new_position", 0)
            and (comment.get("original_position") or 0) == payload.get("old_position", 0)
        ]
        if not originals:
            return self._resp(422, {"message": "unknown review anchor"})
        original = min(originals, key=lambda comment: comment["id"])
        comment_id = self._next_comment_id
        self._next_comment_id += 1
        created = {
            "id": comment_id,
            "pull_request_review_id": review_id,
            "path": payload.get("path", ""),
            "commit_id": original["commit_id"],
            "original_commit_id": original["original_commit_id"],
            "position": payload.get("new_position"),
            "original_position": payload.get("old_position"),
            "extra_lines_count": payload.get("extra_lines_count", 0),
            "user": {"username": self.username},
            "resolver": None,
            "body": payload.get("body", ""),
            "diff_hunk": "@@ -1,3 +1,4 @@",
            "html_url": f"{BASE}/o/r/pulls/1#discussion-{comment_id}",
        }
        self.review_comments.setdefault(review_id, [[]])[-1].append(created)
        self.posts.append((str(request.url), payload))
        return self._resp(201, created)

    # ── assertions helpers ──────────────────────────────────────────

    @property
    def post_urls(self) -> list[str]:
        return [url for method, url, _ in self.requests if method == "POST"]

    def assert_no_issue_comment_traffic(self) -> None:
        assert not any("/issues/" in url for _, url, _ in self.requests)


@pytest.fixture
def server(monkeypatch):
    srv = FakeForgejo()
    monkeypatch.setenv("MIRA_FORGEJO_API_URL", API)
    monkeypatch.delenv("MIRA_FORGEJO_BASE_URL", raising=False)
    monkeypatch.delenv("MIRA_PLATFORMS_JSON_PATH", raising=False)
    profiles._load.cache_clear()
    client_type = httpx.AsyncClient
    monkeypatch.setattr(
        "mira.providers.forgejo.httpx.AsyncClient",
        lambda **kwargs: client_type(transport=httpx.MockTransport(srv.handler), **kwargs),
    )
    yield srv
    profiles._load.cache_clear()


def make_provider() -> ForgejoProvider:
    return ForgejoProvider("fake-token")


def pr(head_sha: str = HEAD_SHA) -> PRInfo:
    return PRInfo(
        title="Synthetic",
        description="",
        base_branch="main",
        head_branch="feature",
        url=f"{BASE}/o/r/pulls/1",
        number=1,
        owner="o",
        repo="r",
        head_sha=head_sha,
        platform="forgejo",
    )


def review_json(rid=500, user=BOT, stale=False, commit="c0ffee"):
    return {
        "id": rid,
        "user": {"username": user},
        "state": "COMMENTED",
        "body": "",
        "commit_id": commit,
        "stale": stale,
        "comments_count": 0,
        "html_url": f"{BASE}/o/r/pulls/1#review-{rid}",
    }


def comment_json(
    cid,
    rid=500,
    path="src/app.py",
    position: int | None = 10,
    original_position: int | None = 0,
    extra_lines_count=0,
    user=BOT,
    commit="c0ffee",
    resolver=None,
    body="**Bug** something is wrong here",
):
    """One review comment as Forgejo's GET returns it (PullReviewComment)."""
    return {
        "id": cid,
        "pull_request_review_id": rid,
        "path": path,
        "commit_id": commit,
        "original_commit_id": commit,
        "position": position,
        "original_position": original_position,
        "extra_lines_count": extra_lines_count,
        "user": {"username": user},
        "resolver": {"username": "alice"} if resolver else None,
        "body": body,
        "diff_hunk": "@@ -1,3 +1,4 @@",
        "html_url": f"{BASE}/o/r/pulls/1#discussion-{cid}",
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z",
    }


def marker_comment_json(cid, target, sha, user=BOT, **anchor):
    """A comment whose body carries the acknowledgment marker for ``target``."""
    body = (
        "✅ **Resolved**\n\n"
        f"Checked at [`{sha[:8]}`]({BASE}/o/r/src/commit/{sha}).\n\n"
        f"<!-- mira:resolved original={target} sha={sha} -->"
    )
    return comment_json(cid, user=user, body=body, **anchor)


class FakeLLM:
    def __init__(self, results):
        self.results = results
        self.calls: list = []

    async def complete(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        return json.dumps({"results": self.results})


def _single_finding(server, **overrides):
    """One bot review with one finding (id 101); returns the comment."""
    comment = comment_json(101, **overrides)
    server.reviews = [[review_json()]]
    server.review_comments = {500: [[comment]]}
    return comment


# ── resolve_threads: transport, anchoring, wording ─────────────────


@pytest.mark.asyncio
async def test_resolve_posts_anchored_reply_for_fixed_finding_only(server):
    _single_finding(server)
    server.review_comments[500][0].append(comment_json(102, position=30, body="**Second** issue"))

    count = await make_provider().resolve_threads(pr(), ["101"])

    assert count == 1
    assert len(server.posts) == 1  # only the fixed finding is acknowledged
    url, payload = server.posts[0]
    assert url == f"{API}/repos/o/r/pulls/1/reviews/500/comments"
    # GET position/original_position/extra_lines_count copied to the POST form.
    assert payload["path"] == "src/app.py"
    assert payload["new_position"] == 10
    assert payload["old_position"] == 0
    assert payload["extra_lines_count"] == 0
    # Wording: resolved + checked-at link, no claim that the SHA introduced the fix.
    body = payload["body"]
    assert body.startswith("✅ **Resolved**")
    assert "Checked at" in body
    assert f"{BASE}/o/r/src/commit/{HEAD_SHA}" in body
    assert "Verified fixed" not in body and "introduced" not in body
    # Machine-readable marker records the original comment ID and checked SHA.
    assert _resolution_marker(body) == (101, HEAD_SHA)
    # Created comment was read back before counting success.
    assert any(
        m == "GET" and u.endswith("/reviews/500/comments/9000") for m, u, _ in server.requests
    )
    server.assert_no_issue_comment_traffic()


@pytest.mark.asyncio
async def test_resolve_sends_fake_token_auth_header(server):
    _single_finding(server)
    await make_provider().resolve_threads(pr(), ["101"])
    posts = [(m, u, h) for m, u, h in server.requests if m == "POST"]
    assert len(posts) == 1
    assert posts[0][2]["authorization"] == "token fake-token"


@pytest.mark.asyncio
async def test_resolve_counts_accurately_when_one_id_is_unknown(server):
    _single_finding(server)
    server.review_comments[500][0].append(comment_json(102, position=30))

    count = await make_provider().resolve_threads(pr(), ["101", "102", "999"])

    assert count == 2  # the unknown ID is a failure, never a fallback post
    assert len(server.posts) == 2
    server.assert_no_issue_comment_traffic()


@pytest.mark.asyncio
async def test_native_resolver_marks_resolved_without_posting(server):
    _single_finding(server, resolver=True)

    provider = make_provider()
    count = await provider.resolve_threads(pr(), ["101"])
    assert count == 1
    assert server.posts == []

    records = await make_provider().get_all_bot_threads(pr(), BOT)
    assert [(r.thread_id, r.is_resolved) for r in records] == [("101", True)]


@pytest.mark.asyncio
async def test_old_side_and_range_preserved_on_stale_deleted_anchor(server):
    # Stale review, deleted file, old-side comment with a 3-line range.
    server.reviews = [[review_json(stale=True)]]
    server.review_comments = {
        500: [
            [
                comment_json(
                    101,
                    path="src/gone.py",
                    position=None,
                    original_position=12,
                    extra_lines_count=3,
                    commit="oldc0de",
                )
            ]
        ]
    }

    count = await make_provider().resolve_threads(pr(), ["101"])

    assert count == 1
    url, payload = server.posts[0]
    assert url == f"{API}/repos/o/r/pulls/1/reviews/500/comments"
    assert payload["path"] == "src/gone.py"
    assert payload["new_position"] == 0  # not on the new side
    assert payload["old_position"] == 12  # side + range preserved
    assert payload["extra_lines_count"] == 3
    server.assert_no_issue_comment_traffic()


# ── get_all_bot_threads: identity, positions, acknowledgments ──────


@pytest.mark.asyncio
async def test_threads_expose_comment_id_position_and_outdated(server):
    _single_finding(server, position=None, original_position=21)
    server.reviews = [[review_json(stale=True)]]

    records = await make_provider().get_all_bot_threads(pr(), BOT)

    assert len(records) == 1
    record = records[0]
    assert record.thread_id == "101"  # original comment ID, not the review ID
    assert record.line == 21  # position falls back to original_position
    assert record.is_resolved is False
    assert record.is_outdated is True
    (unresolved,) = await make_provider().get_unresolved_bot_threads(pr(), BOT)
    assert unresolved.is_outdated is True


@pytest.mark.asyncio
async def test_rerun_on_new_sha_never_duplicates_acknowledgment(server):
    _single_finding(server)
    server.review_comments[500][0].append(marker_comment_json(900, 101, OLD_SHA))

    provider = make_provider()
    records = await provider.get_all_bot_threads(pr(), BOT)
    assert [(r.thread_id, r.is_resolved) for r in records] == [("101", True)]
    assert await provider.get_unresolved_bot_threads(pr(), BOT) == []

    # Rerun on a newer SHA: one acknowledgment per original, not per SHA.
    count = await make_provider().resolve_threads(pr(head_sha="ef" * 20), ["101"])
    assert count == 1
    assert server.posts == []


@pytest.mark.asyncio
async def test_new_acknowledgment_is_recognized_on_next_review(server):
    _single_finding(server)
    provider = make_provider()

    assert await provider.resolve_threads(pr(), ["101"]) == 1
    records = await make_provider().get_all_bot_threads(pr(), BOT)
    assert [(r.thread_id, r.is_resolved) for r in records] == [("101", True)]
    assert await make_provider().get_unresolved_bot_threads(pr(), BOT) == []
    assert await make_provider().resolve_threads(pr(head_sha="ef" * 20), ["101"]) == 1
    assert len(server.posts) == 1


@pytest.mark.asyncio
async def test_second_page_acknowledgment_is_found(server):
    # The acknowledgment lives on the second page of the review's comments.
    server.reviews = [[review_json()]]
    server.review_comments = {500: [[comment_json(101)], [marker_comment_json(900, 101, OLD_SHA)]]}

    records = await make_provider().get_all_bot_threads(pr(), BOT)
    assert records[0].is_resolved is True
    assert await make_provider().resolve_threads(pr(), ["101"]) == 1
    assert server.posts == []


@pytest.mark.asyncio
async def test_human_spoofed_marker_cannot_resolve(server):
    _single_finding(server)
    stolen = marker_comment_json(901, 101, OLD_SHA)
    stolen["user"] = {"username": "mallory"}
    server.review_comments[500][0].append(stolen)

    records = await make_provider().get_all_bot_threads(pr(), BOT)
    assert records[0].is_resolved is False  # not a trusted bot

    assert await make_provider().resolve_threads(pr(), ["101"]) == 1
    assert len(server.posts) == 1


@pytest.mark.asyncio
async def test_marker_at_another_anchor_cannot_resolve(server):
    _single_finding(server)
    server.review_comments[500][0].append(marker_comment_json(902, 101, OLD_SHA, path="other.py"))

    records = await make_provider().get_all_bot_threads(pr(), BOT)
    assert records[0].is_resolved is False
    assert await make_provider().resolve_threads(pr(), ["101"]) == 1
    assert len(server.posts) == 1
    # The mis-anchored marker comment itself is never a finding.
    assert all(r.thread_id != "902" for r in records)


@pytest.mark.asyncio
async def test_marker_targeting_another_comment_cannot_resolve(server):
    _single_finding(server)
    server.review_comments[500][0].append(comment_json(102, position=30))
    server.review_comments[500][0].append(marker_comment_json(903, 999, OLD_SHA))

    records = await make_provider().get_all_bot_threads(pr(), BOT)
    resolved = {r.thread_id: r.is_resolved for r in records}
    assert resolved == {"101": False, "102": False}


@pytest.mark.asyncio
async def test_marker_on_earlier_comment_cannot_resolve(server):
    _single_finding(server)
    # A marker comment created *before* the finding it claims to resolve.
    server.review_comments[500][0].append(marker_comment_json(50, 101, OLD_SHA))

    records = await make_provider().get_all_bot_threads(pr(), BOT)
    assert [r.thread_id for r in records] == ["101"]  # earlier marker is not a finding
    assert records[0].is_resolved is False
    assert await make_provider().resolve_threads(pr(), ["101"]) == 1
    assert len(server.posts) == 1


@pytest.mark.asyncio
async def test_bot_qa_reply_is_not_a_new_finding(server):
    _single_finding(server)
    # Bot answered a human question in the same conversation (no marker).
    server.review_comments[500][0].append(
        comment_json(102, body="Answer: that parameter is validated upstream.")
    )

    records = await make_provider().get_all_bot_threads(pr(), BOT)
    assert [r.thread_id for r in records] == ["101"]  # only the original finding


# ── get_thread_id_for_comment: exact conversation identity ─────────


@pytest.mark.asyncio
async def test_reply_maps_to_original_bot_finding(server):
    _single_finding(server)
    server.review_comments[500][0].append(
        comment_json(200, user="alice", body="Why is this a problem?")
    )

    assert await make_provider().get_thread_id_for_comment("200", pr()) == "101"
    assert await make_provider().get_thread_id_for_comment("777", pr()) is None


@pytest.mark.asyncio
async def test_reply_in_human_conversation_maps_to_nothing(server):
    _single_finding(server)
    server.reviews[0].append(review_json(502, user="alice"))
    server.review_comments[502] = [[comment_json(300, rid=502, user="alice", body="human note")]]

    assert await make_provider().get_thread_id_for_comment("300", pr()) is None


@pytest.mark.asyncio
async def test_same_anchor_ambiguity_is_conservative(server):
    _single_finding(server)
    # Two bot findings at the exact same anchor: no safe mapping exists.
    server.review_comments[500][0].append(comment_json(102, body="**Another** issue"))
    server.review_comments[500][0].append(comment_json(200, user="alice", body="which one?"))

    assert await make_provider().get_thread_id_for_comment("200", pr()) is None
    # Listing stays deterministic: the earliest comment is the conversation's finding.
    records = await make_provider().get_all_bot_threads(pr(), BOT)
    assert [r.thread_id for r in records] == ["101"]


# ── identity and precondition guards ───────────────────────────────


@pytest.mark.asyncio
async def test_missing_identity_blocks_writes_without_any_post(server):
    server.user_status = 500
    _single_finding(server)

    with pytest.raises(ProviderError, match="identity"):
        await make_provider().resolve_threads(pr(), ["101"])
    with pytest.raises(ProviderError, match="identity"):
        await make_provider().reply_to_review_comment(pr(), 101, "hi")

    assert server.posts == []
    assert all(u.endswith("/user") for _, u, _ in server.requests)


@pytest.mark.asyncio
async def test_unknown_identity_never_means_everyone(server):
    server.user_status = 500
    _single_finding(server)

    assert await make_provider().get_all_bot_threads(pr(), None) == []
    assert await make_provider().get_human_review_comments(pr(), "") == []


@pytest.mark.asyncio
async def test_identity_accepts_login_field(server):
    server.user_payload = {"login": BOT}  # no "username" key
    _single_finding(server)

    records = await make_provider().get_all_bot_threads(pr())
    assert [r.thread_id for r in records] == ["101"]
    assert await make_provider().resolve_threads(pr(), ["101"]) == 1
    assert len(server.posts) == 1


@pytest.mark.asyncio
async def test_no_head_sha_performs_no_mutation(server):
    _single_finding(server)
    with pytest.raises(ProviderError, match="head SHA"):
        await make_provider().resolve_threads(pr(head_sha=""), ["101"])
    assert server.posts == []


@pytest.mark.asyncio
async def test_empty_thread_ids_touch_nothing(server):
    assert await make_provider().resolve_threads(pr(), []) == 0
    assert server.requests == []


# ── fail-closed reads, per-ID failures, read-back ──────────────────


@pytest.mark.asyncio
async def test_incomplete_read_fails_closed_before_any_write(server):
    _single_finding(server)
    server.fail_comment_fetch = {500}

    with pytest.raises(ProviderError):
        await make_provider().resolve_threads(pr(), ["101"])
    with pytest.raises(ProviderError):
        await make_provider().get_all_bot_threads(pr(), BOT)
    with pytest.raises(ProviderError):
        await make_provider().reply_to_review_comment(pr(), 101, "hi")
    # The lookup API is best-effort but still returns no guess, never a fallback.
    assert await make_provider().get_thread_id_for_comment("101", pr()) is None
    assert server.posts == []


@pytest.mark.asyncio
async def test_non_list_page_fails_closed(server):
    _single_finding(server)
    server.nonlist_reviews = True
    with pytest.raises(ProviderError):
        await make_provider().resolve_threads(pr(), ["101"])
    assert server.posts == []


@pytest.mark.asyncio
async def test_per_id_post_failure_continues_and_counts(server):
    server.reviews = [[review_json(500), review_json(501)]]
    server.review_comments = {
        500: [[comment_json(101, rid=500)]],
        501: [[comment_json(201, rid=501, position=40)]],
    }
    server.fail_post = {500}

    count = await make_provider().resolve_threads(pr(), ["101", "201"])

    assert count == 1  # first failed, second succeeded — counted accurately
    assert len(server.post_attempts) == 2  # both were attempted; the first 500-ed
    assert len(server.posts) == 1  # only the successful one created a comment
    assert server.posts[0][0].endswith("/reviews/501/comments")


@pytest.mark.asyncio
async def test_readback_failure_is_not_counted_and_never_retried(server):
    _single_finding(server)
    server.fail_readback = {9000}  # the id the POST will create

    count = await make_provider().resolve_threads(pr(), ["101"])

    assert count == 0
    assert len(server.posts) == 1  # no automatic retry that could duplicate


@pytest.mark.asyncio
async def test_readback_with_tampered_body_is_not_counted(server):
    _single_finding(server)
    server.readback_override[9000] = comment_json(9000, body="defaced body")

    assert await make_provider().resolve_threads(pr(), ["101"]) == 0
    assert len(server.posts) == 1


# ── reply_to_review_comment: native API, no top-level fallback ─────


@pytest.mark.asyncio
async def test_reply_uses_original_review_comments_url(server):
    _single_finding(server)

    await make_provider().reply_to_review_comment(pr(), 101, "That is validated upstream.")

    assert len(server.posts) == 1
    url, payload = server.posts[0]
    assert url == f"{API}/repos/o/r/pulls/1/reviews/500/comments"
    assert payload["body"] == "That is validated upstream."
    assert payload["path"] == "src/app.py"
    assert payload["new_position"] == 10
    assert payload["old_position"] == 0
    server.assert_no_issue_comment_traffic()


@pytest.mark.asyncio
async def test_reply_to_unknown_comment_fails_without_fallback(server):
    _single_finding(server)

    with pytest.raises(ProviderError, match="not found"):
        await make_provider().reply_to_review_comment(pr(), 999, "hello?")

    assert server.posts == []
    server.assert_no_issue_comment_traffic()


# ── human learning input: acknowledgments never count ──────────────


@pytest.mark.asyncio
async def test_acks_and_bot_text_never_enter_human_learning(server):
    _single_finding(server)
    server.review_comments[500][0].append(marker_comment_json(900, 101, OLD_SHA))
    server.reviews[0].append(review_json(502, user="alice"))
    server.review_comments[502] = [
        [
            comment_json(
                300,
                rid=502,
                user="alice",
                position=5,
                original_position=0,
                body="please use a constant here",
            )
        ]
    ]

    human = await make_provider().get_human_review_comments(pr(), BOT)

    assert [(h.author, h.body) for h in human] == [("alice", "please use a constant here")]


# ── shared routine end-to-end (real provider, fake model) ──────────


def _seed_two_findings(server):
    server.reviews = [[review_json()]]
    server.review_comments = {
        500: [
            [
                comment_json(101),
                comment_json(102, position=30, body="**Leak** in loop"),
            ]
        ]
    }
    server.contents = {"src/app.py": "safe_code()\n"}


@pytest.mark.asyncio
async def test_resolve_verified_threads_end_to_end(server):
    _seed_two_findings(server)
    llm = FakeLLM([{"id": "101", "fixed": True}, {"id": "102", "fixed": False}])

    checked, resolved, remaining, decisions = await resolve_verified_threads(
        make_provider(),
        llm,
        pr(),
        BOT,
        dry_run=False,  # type: ignore[arg-type]
    )

    assert (checked, resolved) == (2, 1)
    assert [t.thread_id for t in remaining] == ["102"]
    assert [(d.thread_id, d.fixed) for d in decisions] == [("101", True), ("102", False)]
    # Only the fixed finding gets one acknowledgment, anchored to its comment.
    assert len(server.posts) == 1
    _url, payload = server.posts[0]
    assert _resolution_marker(payload["body"]) == (101, HEAD_SHA)
    assert payload["new_position"] == 10
    # The fix check reads the pinned head SHA, not the moving branch name.
    contents = [u for m, u, _ in server.requests if "/contents/" in u]
    assert contents and all(f"ref={HEAD_SHA}" in u for u in contents)
    server.assert_no_issue_comment_traffic()


@pytest.mark.asyncio
async def test_resolve_verified_threads_dry_run_posts_nothing(server):
    _seed_two_findings(server)
    llm = FakeLLM([{"id": "101", "fixed": True}, {"id": "102", "fixed": False}])

    checked, resolved, remaining, _decisions = await resolve_verified_threads(
        make_provider(),
        llm,
        pr(),
        BOT,
        dry_run=True,  # type: ignore[arg-type]
    )

    assert (checked, resolved) == (2, 1)
    assert len(remaining) == 1
    assert server.posts == []
    assert not any(m == "POST" for m, _, _ in server.requests)
    assert len(llm.calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "anchor",
    [
        {"path": ""},
        {"position": 0},
        {"position": -1},
        {"original_position": 10},
        {"extra_lines_count": -1},
    ],
)
async def test_invalid_anchor_never_posts(server, anchor):
    _single_finding(server, **anchor)
    assert await make_provider().resolve_threads(pr(), ["101"]) == 0
    with pytest.raises(ProviderError, match="anchor"):
        await make_provider().reply_to_review_comment(pr(), 101, "reply")
    assert server.posts == []
    assert server.post_attempts == []


@pytest.mark.asyncio
async def test_duplicate_requested_ids_post_once(server):
    _single_finding(server)
    assert await make_provider().resolve_threads(pr(), ["101", "101"]) == 1
    assert len(server.posts) == 1


@pytest.mark.asyncio
async def test_readback_at_wrong_anchor_is_not_success(server):
    _single_finding(server)
    server.readback_override[9000] = marker_comment_json(9000, 101, HEAD_SHA, path="other.py")
    assert await make_provider().resolve_threads(pr(), ["101"]) == 0


@pytest.mark.asyncio
async def test_login_only_review_and_comment_authors(server):
    original = _single_finding(server)
    server.user_payload = {"login": BOT}
    server.reviews[0][0]["user"] = {"login": BOT}
    original["user"] = {"login": BOT}
    assert await make_provider().resolve_threads(pr(), ["101"]) == 1
    assert (await make_provider().get_all_bot_threads(pr()))[0].is_resolved


@pytest.mark.asyncio
async def test_followup_bot_comment_is_not_an_original_finding(server):
    _single_finding(server)
    server.review_comments[500][0].append(comment_json(102, body="An explanation."))
    assert await make_provider().resolve_threads(pr(), ["102"]) == 0
    assert server.posts == []


@pytest.mark.asyncio
async def test_marker_from_different_commit_does_not_resolve(server):
    _single_finding(server)
    acknowledgment = marker_comment_json(900, 101, OLD_SHA)
    acknowledgment["commit_id"] = "different"
    server.review_comments[500][0].append(acknowledgment)
    records = await make_provider().get_all_bot_threads(pr())
    assert [(r.thread_id, r.is_resolved) for r in records] == [("101", False)]


@pytest.mark.asyncio
async def test_later_finding_cannot_be_original_of_earlier_reply(server):
    _single_finding(server)
    server.review_comments[500][0].append(comment_json(50, user="alice"))
    assert await make_provider().get_thread_id_for_comment("50", pr()) is None


@pytest.mark.asyncio
async def test_reply_context_keeps_finding_id_after_resolution_acknowledgment(server):
    _single_finding(server)
    server.review_comments[500][0].append(comment_json(200, user="alice", body="A question."))
    provider = make_provider()
    context = await provider.get_review_reply_context(pr(), 200, BOT)
    assert context is not None
    assert context.thread_id == "101"
    assert await provider.resolve_threads(pr(), [context.thread_id]) == 1

    server.review_comments[500][0].append(comment_json(10000, user="alice", body="A later reply."))
    context = await provider.get_review_reply_context(pr(), 10000, BOT)
    assert context is not None
    assert context.thread_id == "101"
