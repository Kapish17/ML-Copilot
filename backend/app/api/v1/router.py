"""Aggregation of the v1 routers.

Feature routers are registered here and mounted onto the application once, so
the URL prefix lives in a single place.
"""

from __future__ import annotations

from fastapi import APIRouter

from app.api.v1.datasets import router as datasets_router
from app.api.v1.experiments import router as experiments_router
from app.api.v1.knowledge import router as knowledge_router

API_V1_PREFIX = "/api/v1"

api_router = APIRouter(prefix=API_V1_PREFIX)
api_router.include_router(datasets_router)
api_router.include_router(experiments_router)
api_router.include_router(knowledge_router)
