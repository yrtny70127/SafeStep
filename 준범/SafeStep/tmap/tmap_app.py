"""
시각장애인 보행자 네비게이션 - T-map API
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import requests
from gtts import gTTS
import io
import base64

# ==================== API 키 설정 ====================
# 여기에 T-map API 키를 붙여넣으세요
TMAP_API_KEY = "uJiu58Nc89agazFFxqM3lM4u2zv0qtp8XJbQpf9h"

# ==================== FastAPI 앱 ====================

app = FastAPI(title="보행자 네비게이션")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== 모델 ====================

class PlaceSearchRequest(BaseModel):
    query: str

class RouteRequest(BaseModel):
    start: str
    end: str

# ==================== 유틸리티 함수 ====================

def search_place(query: str) -> list:
    """T-map POI 검색"""
    url = "https://apis.openapi.sk.com/tmap/pois"
    
    headers = {
        "Accept": "application/json",
        "appKey": TMAP_API_KEY
    }
    
    params = {
        "version": "1",
        "searchKeyword": query,
        "resCoordType": "WGS84GEO",
        "reqCoordType": "WGS84GEO",
        "count": 10
    }
    
    try:
        print(f"[검색] '{query}'")
        response = requests.get(url, headers=headers, params=params)
        
        if response.status_code == 200:
            data = response.json()
            pois = data.get("searchPoiInfo", {}).get("pois", {}).get("poi", [])
            
            results = []
            for poi in pois[:5]:
                new_addr = poi.get("newAddressList", {}).get("newAddress", [])
                address = new_addr[0].get("fullAddressRoad", "") if new_addr else ""
                
                if not address:
                    address = f"{poi.get('upperAddrName', '')} {poi.get('middleAddrName', '')}".strip()
                
                results.append({
                    "name": poi.get("name"),
                    "address": address,
                    "lat": float(poi.get("noorLat", 0)),
                    "lng": float(poi.get("noorLon", 0))
                })
            
            print(f"  → {len(results)}개 결과 반환")
            return results
        else:
            print(f"  → 오류 {response.status_code}")
            return []
            
    except Exception as e:
        print(f"  → 예외: {e}")
        return []


def get_walking_route(start_lat: float, start_lng: float, end_lat: float, end_lng: float) -> dict:
    """T-map 보행자 경로"""
    url = "https://apis.openapi.sk.com/tmap/routes/pedestrian"
    
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "appKey": TMAP_API_KEY
    }
    
    payload = {
        "startX": str(start_lng),
        "startY": str(start_lat),
        "endX": str(end_lng),
        "endY": str(end_lat),
        "reqCoordType": "WGS84GEO",
        "resCoordType": "WGS84GEO",
        "startName": "출발",
        "endName": "도착"
    }
    
    try:
        print(f"\n[경로 탐색]")
        print(f"  출발: ({start_lat:.6f}, {start_lng:.6f})")
        print(f"  도착: ({end_lat:.6f}, {end_lng:.6f})")
        
        response = requests.post(url, headers=headers, json=payload)
        
        if response.status_code == 200:
            data = response.json()
            features = data.get("features", [])
            
            if not features:
                print("  → 경로를 찾을 수 없습니다\n")
                return {"success": False, "error": "경로 없음"}
            
            total_distance = 0
            total_time = 0
            steps = []
            path_coords = []
            
            for idx, feature in enumerate(features):
                props = feature.get("properties", {})
                geom = feature.get("geometry", {})
                
                # 경로 좌표
                if geom.get("type") == "LineString":
                    for coord in geom.get("coordinates", []):
                        if len(coord) >= 2:
                            # [경도, 위도] → [위도, 경도]
                            path_coords.append([coord[1], coord[0]])
                
                # 총 거리/시간
                if idx == 0:
                    total_distance = props.get("totalDistance", 0)
                    total_time = props.get("totalTime", 0)
                
                # 안내
                desc = props.get("description", "")
                dist = props.get("distance", 0)
                
                if desc:
                    time = int(dist / 1.2) if dist > 0 else 0
                    steps.append({
                        "instruction": desc,
                        "distance": f"{int(dist)}m",
                        "duration": f"{time // 60}분 {time % 60}초" if time >= 60 else f"{time}초"
                    })
            
            print(f"  거리: {total_distance}m")
            print(f"  시간: {total_time}초")
            print(f"  단계: {len(steps)}개")
            print(f"  좌표: {len(path_coords)}개\n")
            
            return {
                "success": True,
                "distance": total_distance,
                "duration": total_time,
                "distance_text": f"{total_distance / 1000:.1f} km",
                "duration_text": f"{total_time // 60}분" if total_time >= 60 else f"{total_time}초",
                "steps": steps,
                "path_coords": path_coords
            }
        else:
            print(f"  → HTTP 오류 {response.status_code}\n")
            return {"success": False, "error": f"API 오류 {response.status_code}"}
            
    except Exception as e:
        print(f"  → 예외: {e}\n")
        return {"success": False, "error": str(e)}


# ==================== API 엔드포인트 ====================

@app.get("/", response_class=HTMLResponse)
async def index():
    """메인 페이지"""
    try:
        with open("tmap_nav.html", 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return """
        <html>
            <body style="font-family: Arial; padding: 50px; text-align: center;">
                <h1>❌ tmap_nav.html 파일이 없습니다</h1>
                <p>같은 폴더에 tmap_nav.html 파일을 넣어주세요.</p>
            </body>
        </html>
        """


@app.post("/api/search")
async def api_search(request: PlaceSearchRequest):
    """장소 검색"""
    if not request.query:
        raise HTTPException(400, "검색어를 입력하세요")
    
    if TMAP_API_KEY == "여기에_TMAP_API_KEY_붙여넣기":
        raise HTTPException(500, "T-map API 키를 설정하세요 (tmap_app.py 13번째 줄)")
    
    results = search_place(request.query)
    return {"success": True, "results": results}


@app.post("/api/directions")
async def api_directions(request: RouteRequest):
    """보행자 경로"""
    if TMAP_API_KEY == "여기에_TMAP_API_KEY_붙여넣기":
        raise HTTPException(500, "T-map API 키를 설정하세요 (tmap_app.py 13번째 줄)")
    
    try:
        start_lat, start_lng = map(float, request.start.split(','))
        end_lat, end_lng = map(float, request.end.split(','))
    except:
        raise HTTPException(400, "좌표 형식 오류 (위도,경도)")
    
    result = get_walking_route(start_lat, start_lng, end_lat, end_lng)
    
    if not result["success"]:
        raise HTTPException(404, result.get("error"))
    
    distance_km = result["distance"] / 1000
    duration_min = result["duration"] / 60
    guidance = f"총 거리 {distance_km:.1f}킬로미터, 예상 시간 {duration_min:.0f}분입니다."
    
    # TTS 음성 생성
    audio_base64 = None
    try:
        tts = gTTS(text=guidance, lang='ko', slow=False)
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        audio_base64 = base64.b64encode(fp.read()).decode('utf-8')
    except Exception as e:
        print(f"TTS 오류: {e}")
    
    return {
        "success": True,
        "distance": result["distance"],
        "duration": result["duration"],
        "distance_text": result["distance_text"],
        "duration_text": result["duration_text"],
        "path_coords": result["path_coords"],
        "steps": result["steps"],
        "guidance": guidance,
        "audio": audio_base64
    }


@app.get("/api/health")
async def health():
    """서버 상태"""
    return {
        "status": "ok",
        "api_ready": TMAP_API_KEY != "여기에_TMAP_API_KEY_붙여넣기"
    }


# ==================== 메인 실행 ====================

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*70)
    print(" " * 20 + "🚶 보행자 네비게이션 (T-map)")
    print("="*70)
    
    if TMAP_API_KEY == "여기에_TMAP_API_KEY_붙여넣기":
        print("\n⚠️  API 키가 설정되지 않았습니다!")
        print("   tmap_app.py 파일의 13번째 줄에서 API 키를 입력하세요")
        print("   TMAP_API_KEY = \"실제_API_키\"")
    else:
        print(f"\n✅ API 키 설정: {TMAP_API_KEY[:10]}...")
    
    print(f"\n🌐 서버 주소: http://localhost:8000")
    print(f"📱 브라우저에서 위 주소를 열어주세요")
    print(f"\n⏹️  종료: Ctrl+C")
    print("="*70 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
