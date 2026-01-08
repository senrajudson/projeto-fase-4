import time

class ResponseTimeMiddleware:
    def init(self, app):
        self.app = app

    async def call(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        start = time.perf_counter()
        status_code = None

        async def send_wrapper(message):
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
            await send(message)

        await self.app(scope, receive, send_wrapper)

        elapsed_ms = (time.perf_counter() - start) * 1000

        method = scope["method"]
        path = scope["path"]

        print(f"{method} {path} -> {status_code} | {elapsed_ms:.2f} ms")

