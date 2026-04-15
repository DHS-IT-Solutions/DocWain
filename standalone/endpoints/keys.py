import time
import uuid

from fastapi import APIRouter, Depends, Header, HTTPException

from standalone.dependencies import get_db
from standalone.auth import generate_api_key, hash_api_key, verify_admin_secret
from standalone.config import Config
from standalone.schemas import KeyCreateRequest, KeyCreateResponse, KeyListItem

router = APIRouter()


def _require_admin(x_admin_secret: str | None = Header(None)):
    if not x_admin_secret:
        raise HTTPException(status_code=401, detail="Missing X-Admin-Secret header")
    if not verify_admin_secret(x_admin_secret, Config.ADMIN_SECRET):
        raise HTTPException(status_code=401, detail="Invalid admin secret")


@router.post("/admin/keys", status_code=201, response_model=KeyCreateResponse)
def create_key(
    body: KeyCreateRequest,
    db=Depends(get_db),
    _=Depends(_require_admin),
):
    raw_key = generate_api_key()
    key_hash = hash_api_key(raw_key)
    key_id = str(uuid.uuid4())
    created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    db.api_keys.insert_one({
        "_id": key_id,
        "key_hash": key_hash,
        "key_prefix": raw_key[:12],
        "name": body.name,
        "active": True,
        "total_requests": 0,
        "created_at": created_at,
    })

    return KeyCreateResponse(
        key_id=key_id,
        raw_key=raw_key,
        key_prefix=raw_key[:12],
        name=body.name,
        created_at=created_at,
    )


@router.get("/admin/keys", response_model=list[KeyListItem])
def list_keys(
    db=Depends(get_db),
    _=Depends(_require_admin),
):
    keys = db.api_keys.find({"active": True})
    return [
        KeyListItem(
            key_id=str(k["_id"]),
            key_prefix=k["key_prefix"],
            name=k["name"],
            created_at=k["created_at"],
            total_requests=k.get("total_requests", 0),
        )
        for k in keys
    ]


@router.delete("/admin/keys/{key_id}")
def delete_key(
    key_id: str,
    db=Depends(get_db),
    _=Depends(_require_admin),
):
    result = db.api_keys.update_one(
        {"_id": key_id, "active": True},
        {"$set": {"active": False}},
    )
    if result.modified_count == 0:
        raise HTTPException(status_code=404, detail="Key not found")
    return {"status": "revoked", "key_id": key_id}
