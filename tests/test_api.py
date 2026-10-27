"""
API Test: Kiểm tra các FastAPI endpoints sử dụng httpx.AsyncClient
"""

import httpx
import pytest

from src.api.main import app


@pytest.mark.asyncio
async def test_root_endpoint():
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "LexiGuard API"
        assert data["status"] == "online"


@pytest.mark.asyncio
async def test_health_endpoint():
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["healthy", "degraded"]
        assert "dependencies" in data
        assert "qdrant" in data["dependencies"]
        assert "neo4j" in data["dependencies"]
        assert "circuit_breaker" in data


@pytest.mark.asyncio
async def test_audit_endpoint_invalid_extension():
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        files = {"file": ("test.exe", b"binarycontent", "application/octet-stream")}
        response = await client.post("/api/v1/audit", files=files)
        assert response.status_code == 400
        assert "Unsupported file format" in response.json()["detail"]


@pytest.mark.asyncio
async def test_audit_endpoint_sse_streaming():
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        contract_content = b"# HOP DONG DICH VU\n\n## Dieu 1: Phat vi pham\nBen A phat Ben B 20% gia tri hop dong neu cham tien do.\n"
        files = {"file": ("contract.md", contract_content, "text/markdown")}
        response = await client.post("/api/v1/audit", files=files)
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]

        body = response.text
        assert "data: " in body
        assert "upload" in body
        assert "completed" in body
