"""
T-map 보행자 길찾기 API
GET  /api/nav/config      → 프론트엔드 SDK 로드용 앱키 반환
POST /api/nav/search      → POI 검색
POST /api/nav/directions  → 보행자 경로 계산
"""
import os
import requests as http
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from backend.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api/nav", tags=["nav"])


def _key():
    k = os.getenv("TMAP_API_KEY", "")
    if not k:
        raise HTTPException(500, "TMAP_API_KEY 환경변수가 설정되지 않았습니다")
    return k


# ── 모델 ──────────────────────────────────────
class SearchReq(BaseModel):
    query: str


class DirectionsReq(BaseModel):
    start_lat: float
    start_lng: float
    end_lat: float
    end_lng: float


# ── 엔드포인트 ────────────────────────────────
@router.get("/config")
def get_config():
    """프론트엔드 T-map JS SDK 로드용 앱키"""
    return {"tmap_key": _key()}


@router.post("/search")
def search_place(req: SearchReq):
    """T-map POI 검색"""
    try:
        resp = http.get(
            "https://apis.openapi.sk.com/tmap/pois",
            headers={"appKey": _key()},
            params={
                "version": "1",
                "searchKeyword": req.query,
                "resCoordType": "WGS84GEO",
                "reqCoordType": "WGS84GEO",
                "count": 7,
            },
            timeout=5,
        )
    except Exception as e:
        raise HTTPException(502, f"T-map 연결 오류: {e}")

    if resp.status_code != 200:
        raise HTTPException(502, f"T-map API 오류: {resp.status_code}")

    pois = resp.json().get("searchPoiInfo", {}).get("pois", {}).get("poi", [])
    results = []
    for poi in pois[:5]:
        new_addr = poi.get("newAddressList", {}).get("newAddress", [])
        address = new_addr[0].get("fullAddressRoad", "") if new_addr else ""
        results.append({
            "name": poi.get("name", ""),
            "address": address,
            "lat": float(poi.get("noorLat", 0)),
            "lng": float(poi.get("noorLon", 0)),
        })
    return {"results": results}


@router.post("/directions")
def get_directions(req: DirectionsReq):
    """T-map 보행자 경로 계산"""
    try:
        resp = http.post(
            "https://apis.openapi.sk.com/tmap/routes/pedestrian",
            headers={"appKey": _key(), "Content-Type": "application/json"},
            json={
                "startX": str(req.start_lng),
                "startY": str(req.start_lat),
                "endX": str(req.end_lng),
                "endY": str(req.end_lat),
                "reqCoordType": "WGS84GEO",
                "resCoordType": "WGS84GEO",
                "startName": "출발",
                "endName": "도착",
            },
            timeout=10,
        )
    except Exception as e:
        raise HTTPException(502, f"T-map 연결 오류: {e}")

    if resp.status_code != 200:
        raise HTTPException(502, f"T-map 경로 오류: {resp.status_code}")

    features = resp.json().get("features", [])
    if not features:
        raise HTTPException(404, "경로를 찾을 수 없습니다")

    total_dist = 0
    total_time = 0
    steps = []
    path_coords = []

    for i, feat in enumerate(features):
        props = feat.get("properties", {})
        geom = feat.get("geometry", {})

        if geom.get("type") == "LineString":
            for coord in geom.get("coordinates", []):
                if len(coord) >= 2:
                    path_coords.append([coord[1], coord[0]])  # [위도, 경도]

        if i == 0:
            total_dist = props.get("totalDistance", 0)
            total_time = props.get("totalTime", 0)

        desc = props.get("description", "")
        dist = props.get("distance", 0)
        if desc:
            steps.append({"instruction": desc, "distance": int(dist)})

    return {
        "total_distance": total_dist,
        "total_time": total_time,
        "distance_text": f"{total_dist / 1000:.1f}km" if total_dist >= 1000 else f"{int(total_dist)}m",
        "duration_text": f"{total_time // 60}분" if total_time >= 60 else f"{total_time}초",
        "steps": steps,
        "path_coords": path_coords,
    }
