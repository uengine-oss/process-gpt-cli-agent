"""Tenant skill management.

Skills are stored CLI-neutral — one folder, one SKILL.md, whatever reference
files travel with it — and converted to each agent's layout only when a run
provisions its workspace. That is why upload here takes no agent argument.
"""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from core.settings import settings
from core.skills import _discover_local_skills


def _tenant_root(tenant_id: str) -> Path:
    root = settings.skills_dirs[0] if settings.skills_dirs else Path("./skills")
    # Tenant skills are namespaced so one tenant's upload cannot shadow
    # another's skill of the same name.
    return root / (tenant_id or "shared")


async def list_skills(request: Request) -> JSONResponse:
    tenant_id = request.query_params.get("tenant_id", "")
    root = _tenant_root(tenant_id)
    tenant_skills = sorted(p.name for p in root.iterdir()) if root.is_dir() else []
    return JSONResponse(
        {
            "tenant_skills": tenant_skills,
            # Bundled skills every tenant gets, listed so the UI can show why a
            # skill is available that nobody uploaded.
            "system_skills": sorted(_discover_local_skills()),
        }
    )


async def upload_skill(request: Request) -> JSONResponse:
    """POST /skills/upload — a .md file or a .zip holding a skill folder."""
    form = await request.form()
    tenant_id = str(form.get("tenant_id") or "")
    upload = form.get("file")
    if upload is None or not hasattr(upload, "filename"):
        return JSONResponse({"error": "file is required"}, status_code=400)

    name = Path(str(upload.filename)).stem
    root = _tenant_root(tenant_id) / name
    root.mkdir(parents=True, exist_ok=True)
    payload = await upload.read()

    if str(upload.filename).lower().endswith(".zip"):
        archive = root / "_upload.zip"
        archive.write_bytes(payload)
        try:
            with zipfile.ZipFile(archive) as bundle:
                # Reject traversal entries rather than trusting the archive.
                for member in bundle.namelist():
                    if member.startswith("/") or ".." in Path(member).parts:
                        return JSONResponse(
                            {"error": f"unsafe path in archive: {member}"}, status_code=400
                        )
                bundle.extractall(root)
        except zipfile.BadZipFile:
            return JSONResponse({"error": "not a valid zip archive"}, status_code=400)
        finally:
            archive.unlink(missing_ok=True)
    else:
        (root / "SKILL.md").write_bytes(payload)

    return JSONResponse({"name": name, "path": str(root)}, status_code=201)


async def delete_skill(request: Request) -> JSONResponse:
    tenant_id = request.query_params.get("tenant_id", "")
    name = request.path_params["name"]
    target = _tenant_root(tenant_id) / name
    if not target.is_dir():
        return JSONResponse({"error": "not found"}, status_code=404)
    shutil.rmtree(target)
    return JSONResponse({"deleted": name})


routes = [
    Route("/skills", list_skills, methods=["GET"]),
    Route("/skills/upload", upload_skill, methods=["POST"]),
    Route("/skills/{name}", delete_skill, methods=["DELETE"]),
]
