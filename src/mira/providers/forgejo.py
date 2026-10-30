"""Forgejo provider using the Gitea-compatible REST API (`/api/v1`).

Mirrors GitHubProvider but speaks Forgejo (Gitea-compatible): pull requests
instead of PRs on GitHub, review comments instead of review threads, and
position-anchored inline review comments. Authentication is a token sent as
``Authorization: token <token>``.

PR URL shape: ``https://forgejo.example.com/owner/repo/pulls/123`` —
note ``/pulls/`` (plural, not GitHub's ``/pull/`` singular).
"""

from __future__ import annotations

import asyncio
import base64
import logging
import re
from collections.abc import AsyncGenerator
from typing import Any
from urllib.parse import quote, urlsplit, urlunsplit

import httpx

from mira.exceptions import ConfigError, ProviderError
from mira.models import (
    BotThreadRecord,
    FileHistoryEntry,
    HumanReviewComment,
    PRInfo,
    ReviewResult,
    UnresolvedThread,
)
from mira.platforms import profiles
from mira.providers.base import CHECK_RUN_NAME, BaseProvider, CheckConclusion
from mira.providers.formatting import format_comment_body, format_key_issues

logger = logging.getLogger(__name__)

# https://forgejo.example.com/owner/repo/pulls/123
_PR_URL_PATTERN = re.compile(
    r"(?:https?://[^/]+/)?(?P<owner>[^/\s]+)/(?P<repo>[^/\s]+?)/pulls/(?P<number>\d+)"
)


def parse_pr_url(pr_url: str) -> tuple[str, str, int]:
    """Parse a PR URL into (owner, repo, number).

    Forgejo ``owner`` is a single segment (no nested groups like GitLab).
    """
    match = _PR_URL_PATTERN.match(pr_url.strip())
    if not match:
        raise ProviderError(
            f"Cannot parse PR URL: {pr_url}. Expected "
            "https://forgejo.example.com/owner/repo/pulls/123"
        )
    return match.group("owner"), match.group("repo"), int(match.group("number"))


def _forgejo_web_base(api_url: str) -> str:
    """Infer the web root only from a standard Forgejo API endpoint."""
    try:
        url = urlsplit(api_url)
        if url.path.endswith("/api/v1"):
            return urlunsplit(url._replace(path=url.path.removesuffix("/api/v1")))
    except ValueError:
        pass
    raise ConfigError(
        "Cannot infer MIRA_FORGEJO_BASE_URL from the Forgejo API URL. "
        "Use an API URL whose path ends in /api/v1, or set "
        "MIRA_FORGEJO_BASE_URL explicitly."
    )


_RESOLUTION_MARKER_RE = re.compile(
    r"(?:^|\n)<!-- mira:resolved original=(\d+) sha=([0-9a-fA-F]{7,64}) -->\s*\Z"
)


def _resolution_marker(body: str | None) -> tuple[int, str] | None:
    """Parse the acknowledgment marker → ``(original_comment_id, checked_sha)``."""
    match = _RESOLUTION_MARKER_RE.search(body or "")
    if not match:
        return None
    return int(match.group(1)), match.group(2).lower()


def _int_or_zero(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _user_login(user: dict[str, Any] | None) -> str:
    user = user or {}
    return str(user.get("login") or user.get("username") or "")


def _comment_line(comment: dict[str, Any]) -> int:
    """GET position is the new-side line; original_position is the old side."""
    return _int_or_zero(comment.get("position") or comment.get("original_position"))


def _conversation_key(review: dict[str, Any], comment: dict[str, Any]) -> tuple[Any, ...]:
    """Match replies to their original review, file, commits, and line side/range."""
    return (
        review.get("id"),
        comment.get("path") or "",
        comment.get("commit_id") or "",
        comment.get("original_commit_id") or "",
        _int_or_zero(comment.get("position")),
        _int_or_zero(comment.get("original_position")),
        _int_or_zero(comment.get("extra_lines_count")),
    )


def _reply_payload(body: str, comment: dict[str, Any]) -> dict[str, Any]:
    """Translate GET line-side fields to the corresponding POST fields."""
    new_line = _int_or_zero(comment.get("position"))
    old_line = _int_or_zero(comment.get("original_position"))
    if (
        not comment.get("path")
        or min(new_line, old_line) < 0
        or (new_line > 0) == (old_line > 0)
        or _int_or_zero(comment.get("extra_lines_count")) < 0
    ):
        raise ProviderError("Cannot reply without an unambiguous review-comment anchor")
    return {
        "body": body,
        "path": comment["path"],
        "new_position": new_line,
        "old_position": old_line,
        "extra_lines_count": _int_or_zero(comment.get("extra_lines_count")),
    }


class ForgejoProvider(BaseProvider):
    """Forgejo code hosting provider (Gitea-compatible REST API)."""

    def __init__(self, token: str) -> None:
        if not token:
            raise ProviderError("Forgejo token is required")
        self._token = token
        profile = profiles.resolve("forgejo")
        self._api = profile["api_url"] or "https://codeberg.org/api/v1"
        self._base = profile.get("base_url") or _forgejo_web_base(self._api.rstrip("/"))
        self._username: str | None = None

    # ── low-level HTTP ──────────────────────────────────────────────

    def _repo(self, pr_info: PRInfo) -> str:
        return f"{self._api}/repos/{quote(pr_info.owner, safe='')}/{quote(pr_info.repo, safe='')}"

    def _pr(self, pr_info: PRInfo) -> str:
        return f"{self._repo(pr_info)}/pulls/{pr_info.number}"

    def _repo_owner(self, pr_info: PRInfo) -> str:
        return f"{self._base}/{quote(pr_info.owner, safe='')}/{quote(pr_info.repo, safe='')}"

    async def _request(
        self, method: str, url: str, *, ok: tuple[int, ...] = (200, 201), **kw: Any
    ) -> httpx.Response:
        headers = {"Authorization": f"token {self._token}", **kw.pop("headers", {})}
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.request(method, url, headers=headers, **kw)
        if resp.status_code not in ok:
            err = ProviderError(f"Forgejo {method} {url} → {resp.status_code}: {resp.text[:300]}")
            err.status_code = resp.status_code  # type: ignore[attr-defined]
            raise err
        return resp

    async def _paginate(self, url: str) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        next_url: str | None = url + ("&" if "?" in url else "?") + "limit=100"
        async with httpx.AsyncClient(timeout=30) as client:
            while next_url:
                resp = await client.get(next_url, headers={"Authorization": f"token {self._token}"})
                resp.raise_for_status()
                page = resp.json()
                if not isinstance(page, list):
                    raise ProviderError(f"Forgejo pagination expected a JSON list from {next_url}")
                out.extend(page)
                next_url = _next_link(resp.headers.get("link", ""))
        return out

    async def _self_username(self) -> str:
        """The token user's own username — who comments are actually posted as."""
        if self._username is None:
            try:
                resp = await self._request("GET", f"{self._api}/user")
                data = resp.json() or {}
                # The token user is identified by either field depending on API shape.
                self._username = _user_login(data)
            except Exception as exc:
                logger.warning("Failed to resolve Forgejo bot identity: %s", exc)
                self._username = ""
        return self._username

    async def _require_write_identity(self) -> str:
        """Require the authenticated identity before writing review replies."""
        username = await self._self_username()
        if not username:
            raise ProviderError(
                "Cannot perform Forgejo write: the authenticated bot identity is "
                "unknown (GET /user failed). Refusing to post as an unidentified user."
            )
        return username

    # ── PR read ─────────────────────────────────────────────────────

    async def get_pr_info(self, pr_url: str) -> PRInfo:
        owner, repo, number = parse_pr_url(pr_url)
        try:
            resp = await self._request(
                "GET",
                f"{self._api}/repos/{quote(owner, safe='')}/{quote(repo, safe='')}/pulls/{number}",
            )
            pr = resp.json()
        except ProviderError:
            raise
        except Exception as e:
            raise ProviderError(f"Failed to fetch PR info: {e}") from e
        return PRInfo(
            title=pr.get("title") or "",
            description=pr.get("body") or "",
            base_branch=(pr.get("base") or {}).get("ref") or "",
            head_branch=(pr.get("head") or {}).get("ref") or "",
            url=pr.get("html_url") or pr_url,
            number=pr.get("number") or number,
            owner=owner,
            repo=repo,
            head_sha=(pr.get("head") or {}).get("sha") or "",
            platform="forgejo",
            author=(pr.get("user") or {}).get("login")
            or (pr.get("user") or {}).get("username", ""),
        )

    async def get_repo_permission(
        self,
        owner: str,
        repo: str,
        username: str,
        *,
        user_id: int | None = None,
    ) -> str | None:
        """Fetch a user's collaborator permission, normalized for authz."""
        url = (
            f"{self._api}/repos/{quote(owner, safe='')}/{quote(repo, safe='')}"
            f"/collaborators/{quote(username, safe='')}/permission"
        )
        try:
            resp = await self._request("GET", url, ok=(200, 404))
        except ProviderError as exc:
            if getattr(exc, "status_code", None) == 403:
                raise ProviderError(
                    f"Forgejo denied the permission lookup for {username!r}. "
                    "The bot account must be a repository admin/owner (or site "
                    "admin) to query other users' permissions; otherwise list "
                    "the user in commands.allowed_users."
                ) from exc
            raise
        except Exception as exc:
            raise ProviderError(f"Failed to fetch repo permission for {username}: {exc}") from exc

        if resp.status_code == 404:
            return "none"
        data = resp.json()
        permission = str((data or {}).get("permission") or "").lower() if data else ""
        if permission in ("admin", "owner", "write"):
            return "write"
        if permission == "read":
            return "read"
        if permission == "none":
            return "none"
        return None

    async def get_pr_diff(self, pr_info: PRInfo) -> str:
        """Fetch the raw unified diff for the PR."""
        try:
            resp = await self._request(
                "GET",
                f"{self._pr(pr_info)}.diff",
                headers={"Accept": "text/plain"},
                ok=(200, 404),
            )
            if resp.status_code == 404:
                logger.warning("PR diff returned 404 for %s", pr_info.url)
                return ""
            return resp.text
        except ProviderError:
            raise
        except Exception as e:
            raise ProviderError(f"Failed to fetch PR diff: {e}") from e

    async def get_compare_diff(self, pr_info: PRInfo, base_sha: str, head_sha: str) -> str:
        """Diff between two commits, for round 2+ incremental reviews."""
        if base_sha == head_sha or not base_sha or not head_sha:
            return ""
        url = (
            f"{self._repo_owner(pr_info)}/compare/{quote(base_sha, safe='')}"
            f"..."
            f"{quote(head_sha, safe='')}.diff"
        )
        try:
            resp = await self._request("GET", url, headers={"Accept": "text/plain"}, ok=(200, 404))
            return resp.text if resp.status_code == 200 else ""
        except ProviderError as exc:
            logger.warning("Compare diff failed: %s", exc)
            return ""
        except Exception as e:
            raise ProviderError(f"Failed to fetch compare diff: {e}") from e

    async def get_file_content(self, pr_info: PRInfo, path: str, ref: str) -> str:
        """Raw file content at a ref — used to verify a thread's fix landed."""
        url = f"{self._repo(pr_info)}/contents/{quote(path, safe='')}?ref={quote(ref, safe='')}"
        try:
            resp = await self._request("GET", url, ok=(200, 404))
        except ProviderError as exc:
            logger.warning("Failed to fetch %s@%s: %s", path, ref, exc)
            return ""
        if resp.status_code == 404:
            return ""
        data = resp.json()
        return base64.b64decode(data.get("content", "")).decode("utf-8", errors="replace")

    async def get_repo_tree(self, pr_info: PRInfo, ref: str) -> list[str]:
        """Every file path in the repo at a ref, for JIT cross-file context.

        Paginates the Gitea git-trees endpoint, which truncates at 1000
        entries per page (``truncated=true``).  Without pagination, large
        repos silently return an incomplete file list.
        """
        base = f"{self._repo(pr_info)}/git/trees/{quote(ref, safe='')}?recursive=true"
        paths: list[str] = []
        page = 1
        try:
            while True:
                url = f"{base}&page={page}" if page > 1 else base
                resp = await self._request("GET", url)
                data = resp.json()
                for entry in data.get("tree") or []:
                    if entry.get("type") == "blob":
                        paths.append(entry["path"])
                if not data.get("truncated", False):
                    break
                page += 1
        except Exception as exc:
            logger.debug("Failed to fetch repo tree: %s", exc)
            return paths
        return paths

    async def get_file_history(
        self, pr_info: PRInfo, paths: list[str], max_per_file: int = 5
    ) -> dict[str, list[FileHistoryEntry]]:
        """Recent commits per file (most-recent first), for decision archaeology."""
        if not paths:
            return {}

        sem = asyncio.Semaphore(8)
        base = f"{self._repo(pr_info)}/commits"
        headers = {"Authorization": f"token {self._token}"}

        async def _fetch_one(
            client: httpx.AsyncClient, path: str
        ) -> tuple[str, list[FileHistoryEntry]]:
            async with sem:
                try:
                    resp = await client.get(
                        base,
                        headers=headers,
                        params={
                            "path": path,
                            "sha": pr_info.head_branch,
                            "limit": max_per_file,
                        },
                    )
                    if resp.status_code != 200:
                        return path, []
                    data = resp.json()
                except Exception as exc:
                    logger.debug("File history fetch failed for %s: %s", path, exc)
                    return path, []

            entries: list[FileHistoryEntry] = []
            for item in data[:max_per_file]:
                message = (item.get("commit") or {}).get("message", "") or ""
                author = (item.get("author") or {}).get("name", "") or ""
                date = (item.get("author") or {}).get("date", "") or ""
                sha = str(item.get("sha", ""))[:8]
                entries.append(
                    FileHistoryEntry(
                        sha=sha,
                        message=message.split("\n\n", 1)[0][:300],
                        author=author,
                        date=date,
                    )
                )
            return path, entries

        async with httpx.AsyncClient(timeout=30) as client:
            results = await asyncio.gather(*[_fetch_one(client, p) for p in paths])
        return {path: hist for path, hist in results if hist}

    # ── posting ─────────────────────────────────────────────────────

    async def post_review(
        self, pr_info: PRInfo, result: ReviewResult, bot_name: str = "miracodeai"
    ) -> None:
        if not result.comments:
            return

        summary_text = ""
        if result.summary:
            summary_text = f"**Mira Review Summary**\n\n{result.summary}"
        if result.key_issues:
            summary_text += format_key_issues(result.key_issues)

        review_body = {
            "event": "COMMENT",
            "body": summary_text or None,
            "commit_id": pr_info.head_sha,
            "comments": [
                {
                    "path": comment.path,
                    "body": format_comment_body(comment, bot_name=bot_name),
                    "new_position": comment.line,
                }
                for comment in result.comments
            ],
        }

        try:
            await self._request("POST", f"{self._pr(pr_info)}/reviews", json=review_body)
        except ProviderError as exc:
            if getattr(exc, "status_code", None) == 422:
                logger.warning("Inline review failed (%s); posting as individual comments", exc)
                if summary_text:
                    try:
                        await self.post_comment(pr_info, summary_text)
                    except ProviderError:
                        logger.warning("Failed to post PR summary comment (fallback)")
                for comment in result.comments:
                    body = format_comment_body(comment, bot_name=bot_name)
                    note = f"**`{comment.path}:{comment.line}`**\n\n{body}"
                    try:
                        await self.post_comment(pr_info, note)
                    except ProviderError:
                        logger.warning(
                            "Plain-comment fallback also failed for %s:%s",
                            comment.path,
                            comment.line,
                        )
            else:
                raise

    async def post_comment(self, pr_info: PRInfo, body: str) -> None:
        await self._request(
            "POST",
            f"{self._repo(pr_info)}/issues/{pr_info.number}/comments",
            json={"body": body},
        )

    # Forgejo has no Checks API, so the commit-status model stands in for it:
    # posting with the same context on the same SHA replaces the state, so one
    # fixed context (``Mira``) renders a single row that transitions
    # pending → success/failure, with the icon following the terminal state.
    # ``start_check_run`` returns ``0`` as a "posted" sentinel —
    # ``finish_check_run`` only skips when it gets ``None``.

    async def start_check_run(
        self,
        pr_info: PRInfo,
        *,
        title: str = "",
        summary: str = "",
        details_url: str = "",
    ) -> int | None:
        """Post a ``pending`` commit status for the PR head commit."""
        if not pr_info.head_sha:
            return None
        payload: dict[str, str] = {
            "state": "pending",
            "context": CHECK_RUN_NAME,
            "description": summary or title,
        }
        if details_url:
            payload["target_url"] = details_url
        await self._request(
            "POST",
            f"{self._repo(pr_info)}/statuses/{quote(pr_info.head_sha, safe='')}",
            json=payload,
        )
        return 0

    async def finish_check_run(
        self,
        pr_info: PRInfo,
        run_id: int | None,
        conclusion: CheckConclusion,
        *,
        title: str = "",
        summary: str = "",
        details: str = "",
    ) -> None:
        """Re-post the commit status with the terminal state for its SHA.

        ``details`` is accepted for interface parity with the Checks API
        path, but commit statuses have no details area.
        """
        if run_id is None or not pr_info.head_sha:
            return
        # Forgejo's commit statuses accept success/failure/error | pending;
        # "neutral" (completed with warnings) renders as a success.
        state = {"success": "success", "failure": "failure", "neutral": "success"}[conclusion]
        payload: dict[str, str] = {
            "state": state,
            "context": CHECK_RUN_NAME,
            "description": summary,
        }
        if pr_info.url:
            payload["target_url"] = pr_info.url
        await self._request(
            "POST",
            f"{self._repo(pr_info)}/statuses/{quote(pr_info.head_sha, safe='')}",
            json=payload,
        )

    async def find_bot_comment(self, pr_info: PRInfo, marker: str) -> int | None:
        try:
            comments = await self._paginate(
                f"{self._repo(pr_info)}/issues/{pr_info.number}/comments"
            )
        except Exception as e:
            raise ProviderError(f"Failed to list issue comments: {e}") from e
        for comment in comments:
            if marker in (comment.get("body") or ""):
                return int(comment["id"])
        return None

    async def update_comment(self, pr_info: PRInfo, comment_id: int, body: str) -> None:
        await self._request(
            "PATCH",
            f"{self._repo(pr_info)}/issues/comments/{comment_id}",
            json={"body": body},
        )

    async def get_comment_body(self, pr_info: PRInfo, comment_id: int) -> str:
        """Fetch an issue comment's body by id. Best-effort."""
        try:
            resp = await self._request("GET", f"{self._repo(pr_info)}/issues/comments/{comment_id}")
            return (resp.json().get("body") or "")[:1500]
        except Exception:
            return ""

    async def reply_to_review_comment(self, pr_info: PRInfo, comment_id: int, body: str) -> None:
        """Reply at the original review anchor, never as a detached issue comment."""
        await self._require_write_identity()
        try:
            pairs = [pair async for pair in self._iter_review_comments(pr_info)]
        except Exception as e:
            raise ProviderError(f"Failed to locate review comment {comment_id}: {e}") from e
        for review, comment in pairs:
            if comment.get("id") == comment_id:
                await self._request(
                    "POST",
                    self._review_comments_url(pr_info, review, comment),
                    json=_reply_payload(body, comment),
                )
                return
        raise ProviderError(
            f"Review comment {comment_id} not found on PR {pr_info.url}; "
            "refusing to post a top-level fallback comment"
        )

    # ── reviews / threads ──────────────────────────────────────────

    async def _fetch_review_comments(self, pr_info: PRInfo, review_id: int) -> list[dict[str, Any]]:
        """Fetch every page of comments for a given review id."""
        return await self._paginate(f"{self._pr(pr_info)}/reviews/{review_id}/comments")

    async def _iter_review_comments(
        self, pr_info: PRInfo
    ) -> AsyncGenerator[tuple[dict[str, Any], dict[str, Any]], None]:
        """Read all pages, propagating failures so hidden replies cannot cause duplicates."""
        for review in await self._paginate(f"{self._pr(pr_info)}/reviews"):
            comments = await self._fetch_review_comments(pr_info, review["id"])
            for comment in comments:
                yield review, comment

    def _review_comments_url(
        self, pr_info: PRInfo, review: dict[str, Any], comment: dict[str, Any]
    ) -> str:
        """The original review's comments endpoint — the anchor replies keep."""
        review_id = review["id"]
        if comment.get("pull_request_review_id", review_id) != review_id:
            raise ProviderError("Review comment belongs to a different review")
        return f"{self._pr(pr_info)}/reviews/{review_id}/comments"

    async def get_all_bot_threads(
        self, pr_info: PRInfo, bot_login: str | None = None
    ) -> list[BotThreadRecord]:
        """Original findings, keyed by comment ID, with native or acknowledged resolutions."""
        bot_identities = {n for n in (await self._self_username(), bot_login) if n}
        if not bot_identities:
            logger.warning(
                "Forgejo bot identity unknown for PR %s; treating no comment as bot-authored",
                pr_info.url,
            )
            return []
        try:
            pairs = [pair async for pair in self._iter_review_comments(pr_info)]
        except Exception as e:
            raise ProviderError(f"Failed to fetch PR reviews: {e}") from e

        findings: dict[tuple[Any, ...], tuple[dict[str, Any], dict[str, Any]]] = {}
        acks: list[tuple[tuple[Any, ...], int, int]] = []
        bot_pairs = [
            (review, comment)
            for review, comment in pairs
            if isinstance(comment.get("id"), int)
            and _user_login(review.get("user")) in bot_identities
            and _user_login(comment.get("user")) in bot_identities
        ]
        for review, comment in sorted(bot_pairs, key=lambda pair: pair[1]["id"]):
            key = _conversation_key(review, comment)
            marker = _resolution_marker(comment.get("body"))
            if marker is not None:
                acks.append((key, marker[0], comment["id"]))
                continue
            findings.setdefault(key, (comment, review))  # earliest = the finding

        resolved_keys = {key for key, (comment, _) in findings.items() if comment.get("resolver")}
        for key, target, ack_id in acks:
            entry = findings.get(key)
            # Trust only a real bot acknowledgment posted *after* the finding
            # it targets, at the finding's exact anchor (key). Earlier or
            # mis-anchored markers, and markers targeting a different
            # comment, resolve nothing.
            if entry is not None and target == entry[0]["id"] and ack_id > entry[0]["id"]:
                resolved_keys.add(key)

        return [
            BotThreadRecord(
                thread_id=str(comment["id"]),
                path=comment.get("path") or "",
                line=_comment_line(comment),
                body=comment.get("body") or "",
                is_resolved=key in resolved_keys,
                is_outdated=bool(review.get("stale")),
            )
            for key, (comment, review) in findings.items()
        ]

    async def get_unresolved_bot_threads(
        self, pr_info: PRInfo, bot_login: str | None = None
    ) -> list[UnresolvedThread]:
        threads = await self.get_all_bot_threads(pr_info, bot_login)
        return [
            UnresolvedThread(
                thread_id=t.thread_id,
                path=t.path,
                line=t.line,
                body=t.body,
                is_outdated=t.is_outdated,
            )
            for t in threads
            if not t.is_resolved
        ]

    async def resolve_threads(self, pr_info: PRInfo, thread_ids: list[str]) -> int:
        """Acknowledge original finding IDs in-thread, reusing existing acknowledgments.

        Forgejo has no native resolve API. The reply records the checked SHA;
        it does not claim that commit introduced the fix. Read before posting
        prevents sequential replays, not simultaneous writers, from duplicating replies.
        """
        if not thread_ids:
            return 0
        username = await self._require_write_identity()
        if not pr_info.head_sha:
            raise ProviderError(
                f"Cannot record resolutions on PR {pr_info.url}: the PR head SHA is "
                "unknown; refusing to acknowledge findings without a checked revision."
            )
        try:
            pairs = [pair async for pair in self._iter_review_comments(pr_info)]
        except Exception as e:
            raise ProviderError(f"Failed to fetch review comments before resolving: {e}") from e

        by_id = {
            comment["id"]: (review, comment)
            for review, comment in pairs
            if isinstance(comment.get("id"), int)
        }
        acks = [
            (_conversation_key(review, comment), marker[0], comment["id"])
            for review, comment in pairs
            if isinstance(comment.get("id"), int) and _user_login(comment.get("user")) == username
            for marker in [_resolution_marker(comment.get("body"))]
            if marker is not None
        ]

        sha = pr_info.head_sha.lower()
        resolved = 0
        for tid in dict.fromkeys(thread_ids):
            try:
                original_id = int(tid)
            except (TypeError, ValueError):
                logger.warning(
                    "Ignoring unresolvable Forgejo thread id %r on PR %s", tid, pr_info.url
                )
                continue
            entry = by_id.get(original_id)
            if entry is None:
                logger.warning(
                    "Unknown Forgejo thread id %s on PR %s; not posting anything for it",
                    tid,
                    pr_info.url,
                )
                continue
            review, comment = entry
            if (
                _user_login(comment.get("user")) != username
                or _user_login(review.get("user")) != username
            ):
                logger.warning(
                    "Refusing to resolve comment %s on PR %s: not authored by the bot (%r)",
                    tid,
                    pr_info.url,
                    username,
                )
                continue
            if comment.get("resolver"):
                resolved += 1  # natively resolved already
                continue
            key = _conversation_key(review, comment)
            if _resolution_marker(comment.get("body")) is not None or any(
                isinstance(other.get("id"), int)
                and other["id"] < original_id
                and _user_login(other.get("user")) == username
                and _resolution_marker(other.get("body")) is None
                and _conversation_key(other_review, other) == key
                for other_review, other in pairs
            ):
                logger.warning("Comment %s is not the original finding; not resolving it", tid)
                continue
            if any(
                ack_key == key and target == original_id and ack_id > original_id
                for ack_key, target, ack_id in acks
            ):
                logger.debug("Forgejo finding %s already acknowledged; not reposting", tid)
                resolved += 1
                continue
            body = (
                "✅ **Resolved**\n\n"
                f"Checked at [`{sha[:8]}`]({self._repo_owner(pr_info)}/src/commit/{sha}).\n\n"
                f"<!-- mira:resolved original={original_id} sha={sha} -->"
            )
            try:
                resp = await self._request(
                    "POST",
                    self._review_comments_url(pr_info, review, comment),
                    json=_reply_payload(body, comment),
                )
                created_id = (resp.json() or {}).get("id")
                if not isinstance(created_id, int):
                    logger.warning(
                        "Resolution reply for %s on PR %s returned no comment id",
                        tid,
                        pr_info.url,
                    )
                    continue
                # Read the created comment back before reporting success; never
                # auto-retry the POST, which could duplicate the acknowledgment.
                back = (
                    await self._request(
                        "GET",
                        f"{self._review_comments_url(pr_info, review, comment)}/{created_id}",
                    )
                ).json() or {}
                if (
                    back.get("id") != created_id
                    or _user_login(back.get("user")) != username
                    or _resolution_marker(back.get("body")) != (original_id, sha)
                    or _conversation_key(review, back) != key
                ):
                    logger.warning(
                        "Resolution reply %s for finding %s failed read-back on PR %s",
                        created_id,
                        tid,
                        pr_info.url,
                    )
                    continue
                resolved += 1
            except Exception as exc:
                logger.warning(
                    "Failed to post resolution reply for comment %s on PR %s: %s",
                    tid,
                    pr_info.url,
                    exc,
                )
        if resolved < len(thread_ids):
            logger.warning(
                "Resolved %d/%d Forgejo finding(s) on PR %s",
                resolved,
                len(thread_ids),
                pr_info.url,
            )
        return resolved

    async def resolve_outdated_review_threads(self, pr_info: PRInfo) -> int:
        """Resolve outdated review threads — no-op for Forgejo."""
        logger.debug("resolve_outdated_review_threads is a no-op for Forgejo")
        return 0

    async def get_thread_id_for_comment(self, comment_node_id: str, pr_info: PRInfo) -> str | None:
        """Map an exact conversation anchor to a unique preceding bot finding."""
        username = await self._self_username()
        if not username:
            return None
        try:
            pairs = [pair async for pair in self._iter_review_comments(pr_info)]
        except Exception as exc:
            logger.warning(
                "Failed to look up thread for comment %s on PR %s: %s",
                comment_node_id,
                pr_info.url,
                exc,
            )
            return None
        target = next((pair for pair in pairs if str(pair[1].get("id")) == comment_node_id), None)
        if target is None:
            return None
        key = _conversation_key(*target)
        candidates = {
            comment["id"]
            for review, comment in pairs
            if isinstance(comment.get("id"), int)
            and _user_login(review.get("user")) == username
            and _user_login(comment.get("user")) == username
            and comment["id"] <= target[1]["id"]
            and _resolution_marker(comment.get("body")) is None
            and _conversation_key(review, comment) == key
        }
        if len(candidates) != 1:
            if candidates:
                logger.warning(
                    "Ambiguous bot findings at the anchor of comment %s on PR %s: %s",
                    comment_node_id,
                    pr_info.url,
                    sorted(candidates),
                )
            return None
        return str(next(iter(candidates)))

    async def get_review_reply_context(
        self, pr_info: PRInfo, incoming_comment_id: int | str, bot_login: str | None = None
    ) -> BotThreadRecord | None:
        """The verified bot original a Forgejo reply comment answers.

        Forgejo's inline-reply webhook is a plain ``issue_comment`` (only
        id/body/user), so the reply's identity must come from the review
        comments API: the incoming id has to exist there as a code comment.
        Forgejo replies are created via ``CreateCodeCommentKnownReviewID``,
        which copies the original's review id, tree path, side/line and
        commits onto the reply — so the original is the earlier bot comment
        at exactly that anchor. Anything but exactly one such candidate is
        a guess and yields ``None`` (never: nearest, latest, or human).
        """
        target = str(incoming_comment_id)
        bot_identities = {n for n in (await self._self_username(), bot_login) if n}
        if not bot_identities:
            logger.warning(
                "Cannot verify bot identity for reply context on %s; refusing to trust a root",
                pr_info.url,
            )
            return None
        try:
            async for review, comment in self._iter_review_comments(pr_info):
                if str(comment.get("id")) != target:
                    continue
                return await self._find_bot_original(pr_info, review, comment, bot_identities)
        except Exception as exc:
            logger.warning(
                "Failed to resolve Forgejo reply context for comment %s on %s: %s",
                target,
                pr_info.url,
                exc,
            )
            return None
        logger.info(
            "Forgejo comment %s not found among %s review comments; no reply context",
            target,
            pr_info.url,
        )
        return None

    async def _find_bot_original(
        self,
        pr_info: PRInfo,
        review: dict[str, Any],
        reply: dict[str, Any],
        bot_identities: set[str],
    ) -> BotThreadRecord | None:
        """The unique earlier bot comment at the reply's exact anchor, or None.

        Anchor = same review id, same path, same position side (``position``
        for new-side, ``original_position`` for old-side — a comment whose
        non-zero line is on the other side never matches), and same
        ``commit_id``/``original_commit_id``. Zero or multiple candidates
        return ``None``; identity is never guessed.
        """
        try:
            reply_id = int(reply.get("id") or 0)
            reply_new = int(reply.get("position") or 0)
            reply_old = int(reply.get("original_position") or 0)
        except (TypeError, ValueError):
            return None
        if not reply_new and not reply_old:
            # No coordinates at all — any match would be a guess.
            return None
        review_id = str(review.get("id"))
        candidates: list[tuple[dict[str, Any], dict[str, Any]]] = []

        async for r, c in self._iter_review_comments(pr_info):
            if str(r.get("id")) != review_id:
                continue
            try:
                cid = int(c.get("id") or 0)
            except (TypeError, ValueError):
                continue
            if cid >= reply_id:
                continue  # must be an *earlier* comment (the original, not the reply)
            if _resolution_marker(c.get("body")) is not None:
                continue
            if (c.get("path") or "") != (reply.get("path") or ""):
                continue
            if reply_new:
                if int(c.get("position") or 0) != reply_new:
                    continue
            else:
                if int(c.get("original_position") or 0) != reply_old:
                    continue
            if (c.get("commit_id") or "") != (reply.get("commit_id") or ""):
                continue
            if (c.get("original_commit_id") or "") != (reply.get("original_commit_id") or ""):
                continue
            if (c.get("user") or {}).get("username", "") not in bot_identities:
                continue
            candidates.append((r, c))

        if len(candidates) != 1:
            logger.info(
                "Forgejo reply %s on %s matched %d bot original(s); requiring exactly 1",
                reply.get("id"),
                pr_info.url,
                len(candidates),
            )
            return None
        original_review, original = candidates[0]
        return BotThreadRecord(
            thread_id=str(original["id"]),
            path=original.get("path") or "",
            line=int(original.get("position") or original.get("original_position") or 0),
            body=original.get("body") or "",
            is_resolved=bool(original.get("resolver")),
            is_outdated=bool(original_review.get("stale")),
        )

    async def get_human_review_comments(
        self, pr_info: PRInfo, bot_login: str
    ) -> list[HumanReviewComment]:
        bot_identities = {n for n in (await self._self_username(), bot_login) if n}
        if not bot_identities:
            logger.warning(
                "Forgejo bot identity unknown for PR %s; not harvesting human comments "
                "(bot-authored text could leak into learning)",
                pr_info.url,
            )
            return []
        out: list[HumanReviewComment] = []

        try:
            pairs = [pair async for pair in self._iter_review_comments(pr_info)]
        except Exception as e:
            raise ProviderError(f"Failed to fetch reviews for human comments: {e}") from e

        for review, comment in pairs:
            review_user = _user_login(review.get("user"))
            if review_user in bot_identities:
                continue

            comment_user = _user_login(comment.get("user"))
            if comment_user in bot_identities:
                continue

            line = _comment_line(comment)
            out.append(
                HumanReviewComment(
                    path=comment.get("path") or "",
                    line=int(line),
                    body=comment.get("body") or "",
                    author=comment_user,
                )
            )

        return out

    # ── labels ──────────────────────────────────────────────────────

    async def get_pr_description(self, pr_info: PRInfo) -> str:
        resp = await self._request("GET", self._pr(pr_info))
        return (resp.json() or {}).get("body") or ""

    async def update_pr_description(self, pr_info: PRInfo, body: str) -> None:
        await self._request("PATCH", self._pr(pr_info), json={"body": body})

    async def add_label(self, pr_info: PRInfo, label: str) -> None:
        await self._request(
            "POST",
            f"{self._repo(pr_info)}/issues/{pr_info.number}/labels",
            json={"labels": [label]},
        )

    async def remove_label(self, pr_info: PRInfo, label: str) -> None:
        # Quoting preserves dot segments, which HTTP clients normalize to a parent resource.
        if label in ("", ".", ".."):
            raise ProviderError("Cannot remove an empty or dot-segment label identifier")
        await self._request(
            "DELETE",
            f"{self._repo(pr_info)}/issues/{pr_info.number}/labels/{quote(label, safe='')}",
            ok=(200, 204),
        )

    async def get_discussion_root_body(self, pr_info: PRInfo, discussion_id: str) -> str:
        """The first comment of a thread/discussion. Best-effort.

        Forgejo has no discussion threading (replies are flat issue comments),
        so this is equivalent to ``get_comment_body``.
        """
        try:
            resp = await self._request(
                "GET",
                f"{self._repo(pr_info)}/issues/comments/{discussion_id}",
            )
            return (resp.json().get("body") or "")[:1500]
        except Exception:
            return ""


def _next_link(link_header: str) -> str | None:
    """Extract the rel="next" URL from a Link header."""
    if not link_header:
        return None
    for part in link_header.split(","):
        if 'rel="next"' in part:
            return part.split(";")[0].strip().strip("<>")
    return None
