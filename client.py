"""Async HTTP client with retry logic for forwarding requests to providers."""

import asyncio
import logging
from typing import Awaitable, Callable

import httpx

logger = logging.getLogger(__name__)

_RETRY_STATUSES = {429, 500, 502, 503, 504}
_CONNECT_TIMEOUT_SEC = 3.0
_MAX_CONNECT_RETRY_DELAY_SEC = 60.0

# Shared HTTP client for non-streaming requests (connection pooling)
_shared_client: httpx.AsyncClient | None = None


def upstream_timeout(timeout: float) -> httpx.Timeout:
    """Use a short TCP connect timeout without affecting established connections."""
    return httpx.Timeout(timeout, connect=_CONNECT_TIMEOUT_SEC)


def get_shared_client() -> httpx.AsyncClient:
    """Return the shared HTTP client (connection pooling). Created lazily."""
    global _shared_client
    if _shared_client is None:
        _shared_client = httpx.AsyncClient(timeout=upstream_timeout(600.0), trust_env=False)
    return _shared_client


async def close_shared_client() -> None:
    """Close the shared client. Called during app shutdown."""
    global _shared_client
    if _shared_client:
        await _shared_client.aclose()
        _shared_client = None


class ProviderError(Exception):
    def __init__(self, status: int, body: str):
        super().__init__(f"Provider returned HTTP {status}: {body[:200]}")
        self.status = status
        self.body = body


class ProviderStream:
    """Async iterable yielding SSE lines from an established provider connection."""

    def __init__(
        self,
        response: httpx.Response,
        client: httpx.AsyncClient,
        reconnect=None,
        max_retries: int = 0,
        disconnect_check: Callable[[], Awaitable[bool]] | None = None,
    ):
        self._response = response
        self._client = client
        self._reconnect = reconnect
        self._max_retries = max_retries
        self._disconnect_check = disconnect_check

    async def __aiter__(self):
        attempt = 0
        yielded_any = False

        while True:
            try:
                async for line in self._response.aiter_lines():
                    if line:
                        yielded_any = True
                        yield line.encode()
                return
            except (httpx.ConnectError, httpx.TimeoutException, httpx.RemoteProtocolError) as exc:
                can_retry = not yielded_any and self._reconnect is not None
                if not can_retry:
                    raise

                attempt += 1
                await self.aclose()
                if attempt <= self._max_retries:
                    logger.warning(
                        "Stream read error before first chunk (attempt %d/%d): %s",
                        attempt,
                        self._max_retries,
                        exc,
                    )
                else:
                    logger.warning(
                        "Stream read error before first chunk after retries exhausted; "
                        "holding request open and retrying (attempt %d): %s",
                        attempt,
                        exc,
                    )
                await _sleep_with_disconnect(
                    _connect_retry_delay(attempt),
                    self._disconnect_check,
                )
                self._response, self._client = await self._reconnect()

    async def aiter_raw(self):
        async for chunk in self._response.aiter_raw():
            if chunk:
                yield chunk

    async def aclose(self):
        await self._response.aclose()
        await self._client.aclose()


async def post_json(
    url: str,
    headers: dict,
    body: dict,
    timeout: float = 600.0,
    max_retries: int = 3,
    disconnect_check: Callable[[], Awaitable[bool]] | None = None,
) -> dict:
    """POST JSON, return parsed response dict. Retries on transient errors."""
    last_exc: Exception | None = None
    client = get_shared_client()

    attempt = 0
    while True:
        if attempt > 0:
            if attempt <= max_retries:
                logger.warning("Retry %d/%d after sleep", attempt, max_retries)
                await asyncio.sleep(1)
            else:
                raise last_exc or ProviderError(0, "Unknown error after retries")

        try:
            resp = await client.post(url, headers=headers, json=body, timeout=upstream_timeout(timeout))
        except (httpx.ConnectError, httpx.TimeoutException, httpx.RemoteProtocolError) as exc:
            last_exc = exc
            logger.warning("Connection error (attempt %d): %s", attempt + 1, exc)
            attempt += 1
            continue

        if resp.status_code in _RETRY_STATUSES and attempt < max_retries:
            logger.warning(
                "HTTP %d from provider (attempt %d), will retry",
                resp.status_code, attempt + 1,
            )
            last_exc = ProviderError(resp.status_code, resp.text)
            attempt += 1
            continue

        if resp.status_code >= 400:
            raise ProviderError(resp.status_code, resp.text)

        return resp.json()


async def open_provider_stream(
    url: str,
    headers: dict,
    body: dict,
    timeout: float = 600.0,
    max_retries: int = 3,
    disconnect_check: Callable[[], Awaitable[bool]] | None = None,
) -> ProviderStream:
    """
    Eagerly open a streaming connection to the provider.
    Returns a ProviderStream (async iterable of SSE lines).
    Raises ProviderError immediately if provider returns an error status.
    Retries on transient errors before raising.
    """
    last_exc: Exception | None = None

    async def _connect() -> tuple[httpx.Response, httpx.AsyncClient]:
        client = httpx.AsyncClient(timeout=upstream_timeout(timeout), trust_env=False)
        req = client.build_request("POST", url, headers=headers, json=body)
        try:
            resp = await client.send(req, stream=True)
        except Exception:
            await client.aclose()
            raise
        return resp, client

    attempt = 0
    while True:
        if attempt > 0:
            if attempt <= max_retries:
                logger.warning("Stream retry %d/%d after sleep", attempt, max_retries)
                await asyncio.sleep(1)
            else:
                raise last_exc or ProviderError(0, "Unknown error after retries")

        try:
            resp, client = await _connect()
        except (httpx.ConnectError, httpx.TimeoutException, httpx.RemoteProtocolError) as exc:
            last_exc = exc
            logger.warning("Stream connection error (attempt %d): %s", attempt + 1, exc)
            attempt += 1
            continue

        if resp.status_code in _RETRY_STATUSES and attempt < max_retries:
            logger.warning(
                "HTTP %d from provider (stream attempt %d), will retry",
                resp.status_code, attempt + 1,
            )
            last_exc = ProviderError(resp.status_code, "")
            await resp.aclose()
            await client.aclose()
            attempt += 1
            continue

        if resp.status_code >= 400:
            body_text = await resp.aread()
            await resp.aclose()
            await client.aclose()
            raise ProviderError(resp.status_code, body_text.decode())

        return ProviderStream(
            resp,
            client,
            reconnect=_connect,
            max_retries=max_retries,
            disconnect_check=disconnect_check,
        )


def _connect_retry_delay(attempt: int) -> float:
    """Return exponential backoff in seconds, capped at one minute."""
    return min(float(2 ** max(attempt - 1, 0)), _MAX_CONNECT_RETRY_DELAY_SEC)


async def _sleep_with_disconnect(
    delay: float,
    disconnect_check: Callable[[], Awaitable[bool]] | None,
) -> None:
    """Sleep in small increments so held requests stop once the client disconnects."""
    if disconnect_check is None:
        await asyncio.sleep(delay)
        return

    remaining = delay
    while remaining > 0:
        if await disconnect_check():
            raise asyncio.CancelledError("Client disconnected while waiting for upstream")
        interval = min(1.0, remaining)
        await asyncio.sleep(interval)
        remaining -= interval
