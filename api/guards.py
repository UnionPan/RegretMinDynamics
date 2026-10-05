"""Bound request buffers and response concurrency through the ASGI lifecycle."""

from fastapi.responses import JSONResponse

from .store import RateLimitExceeded

MAX_BODY_BYTES = 64 * 1024


class RequestBodyLimitMiddleware:
    """Buffer at most 64 KiB before FastAPI parses a POST request body."""

    def __init__(self, app, max_bytes=MAX_BODY_BYTES):
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] != "POST":
            return await self.app(scope, receive, send)
        declared = next(
            (value for key, value in scope["headers"] if key == b"content-length"), None
        )
        try:
            oversized = declared is not None and int(declared) > self.max_bytes
        except ValueError:
            return await JSONResponse(
                status_code=400, content={"detail": "Invalid request length."}
            )(scope, receive, send)
        body = bytearray()
        while not oversized:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            chunk = message.get("body", b"")
            if len(body) + len(chunk) > self.max_bytes:
                oversized = True
                break
            body.extend(chunk)
            if not message.get("more_body", False):
                break
        if oversized:
            return await JSONResponse(
                status_code=413, content={"detail": "Request bodies must be 64 KiB or smaller."}
            )(scope, receive, send)
        buffered = bytes(body)
        delivered = False

        async def replay_body():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": buffered, "more_body": False}
            return await receive()

        return await self.app(scope, replay_body, send)


class RequestBudgetMiddleware:
    def __init__(self, app, limiter):
        self.app = app
        self.limiter = limiter

    async def __call__(self, scope, receive, send):
        path = scope.get("path", "")
        if scope["type"] == "http" and (path == "/api" or path.startswith("/api/")):
            client = scope.get("client")
            kind = "run" if scope["method"] == "POST" and path == "/api/experiments" else "read"
            try:
                self.limiter.check(client[0] if client else "unknown", kind)
            except RateLimitExceeded as error:
                response = JSONResponse(
                    status_code=429,
                    content={"detail": "Request limit reached. Try again shortly."},
                    headers={"Retry-After": str(error.retry_after)},
                )
                return await response(scope, receive, send)
        return await self.app(scope, receive, send)


class ResourceConcurrencyMiddleware:
    """Hold memory-heavy response slots until sending completes or fails."""

    def __init__(self, app, render_slots, archive_slot):
        self.app = app
        self.render_slots = render_slots
        self.archive_slot = archive_slot

    async def __call__(self, scope, receive, send):
        parts = scope.get("path", "").strip("/").split("/")
        slot = None
        if (
            scope["type"] == "http"
            and scope["method"] == "GET"
            and len(parts) == 4
            and parts[:2] == ["api", "experiments"]
        ):
            if parts[-1] == "figure":
                slot = self.render_slots
            elif parts[-1] == "download":
                slot = self.archive_slot
        if slot is None:
            return await self.app(scope, receive, send)
        if not slot.acquire(blocking=False):
            response = JSONResponse(
                status_code=429,
                content={"detail": "Response preparation is busy. Try again shortly."},
                headers={"Retry-After": "2"},
            )
            return await response(scope, receive, send)
        try:
            return await self.app(scope, receive, send)
        finally:
            slot.release()
