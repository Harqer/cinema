"""
dialogflow/client.py
─────────────────────────────────────────────────────────────────────────────
Thin authenticated REST client wrapping the Dialogflow CX v3 API.

Uses google-auth Application Default Credentials (ADC) — whichever identity
`gcloud auth application-default login` or the service account provides.
No hard-coded keys.
"""

from __future__ import annotations

import json
from typing import Any

import google.auth
import google.auth.transport.requests
import requests as http_requests


_API_BASE = "https://dialogflow.googleapis.com/v3"


class DialogflowCXClient:
    """
    Authenticated HTTP client for the Dialogflow CX v3 REST API.

    Usage::

        client = DialogflowCXClient()

        # GET
        data = client.get("projects/my-project/locations/us-central1/agents")

        # POST
        agent = client.post(
            "projects/my-project/locations/us-central1/agents",
            body={"displayName": "cinema", "defaultLanguageCode": "en", ...}
        )
    """

    def __init__(self) -> None:
        self._creds, _ = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        self._session = google.auth.transport.requests.AuthorizedSession(self._creds)

    # ── Low-level helpers ──────────────────────────────────────────────────────

    def _url(self, resource_path: str) -> str:
        """Prepend the API base if the path is not already a full URL."""
        if resource_path.startswith("https://"):
            return resource_path
        path = resource_path.lstrip("/")
        return f"{_API_BASE}/{path}"

    def _raise(self, resp: http_requests.Response) -> None:
        if not resp.ok:
            try:
                detail = resp.json()
            except Exception:
                detail = resp.text
            raise RuntimeError(
                f"Dialogflow CX API error {resp.status_code}: {json.dumps(detail, indent=2)}"
            )

    # ── HTTP verbs ─────────────────────────────────────────────────────────────

    def get(self, resource_path: str, params: dict | None = None) -> dict:
        resp = self._session.get(self._url(resource_path), params=params or {})
        self._raise(resp)
        return resp.json()

    def post(self, resource_path: str, body: dict) -> dict:
        resp = self._session.post(
            self._url(resource_path),
            json=body,
            headers={"Content-Type": "application/json"},
        )
        self._raise(resp)
        return resp.json()

    def patch(self, resource_path: str, body: dict, update_mask: str = "") -> dict:
        params = {"updateMask": update_mask} if update_mask else {}
        resp = self._session.patch(
            self._url(resource_path),
            json=body,
            headers={"Content-Type": "application/json"},
            params=params,
        )
        self._raise(resp)
        return resp.json()

    def delete(self, resource_path: str) -> None:
        resp = self._session.delete(self._url(resource_path))
        self._raise(resp)

    def list_all(self, resource_path: str, key: str, params: dict | None = None) -> list[dict]:
        """
        Paginate through a list endpoint, collecting all items under ``key``.
        Handles nextPageToken automatically.
        """
        results: list[dict] = []
        page_params: dict = dict(params or {})

        while True:
            resp = self._session.get(self._url(resource_path), params=page_params)
            self._raise(resp)
            data = resp.json()
            results.extend(data.get(key, []))
            token = data.get("nextPageToken")
            if not token:
                break
            page_params["pageToken"] = token

        return results
