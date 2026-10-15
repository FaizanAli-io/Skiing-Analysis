from fastapi.middleware.gzip import GZipMiddleware


class ApiGZipMiddleware:
    """Gzip API responses only; videos and files are already compressed and
    must keep byte-range support for streaming."""

    def __init__(self, app, minimum_size: int = 1000):
        self.app = app
        self.gzip = GZipMiddleware(app, minimum_size=minimum_size)

    async def __call__(self, scope, receive, send):
        path = scope.get("path", "") if scope.get("type") == "http" else ""
        if path.startswith("/api/") and not path.endswith("/download"):
            await self.gzip(scope, receive, send)
        else:
            await self.app(scope, receive, send)
