let selectedStart = null;
let selectedEnd = null;
let searchTimeout = null;
let routeData = null;
let currentStepIndex = 0;
let watchId = null;
let map = null;
let marker = null;
let pathData = null;

// 출발지 검색
document.getElementById('start').addEventListener('input', function(e) {
    clearTimeout(searchTimeout);
    const query = e.target.value.trim();
    
    if (query.length < 2) {
        document.getElementById('startResults').classList.remove('show');
        return;
    }
    
    searchTimeout = setTimeout(() => search(query, 'start'), 500);
});

// 목적지 검색
document.getElementById('end').addEventListener('input', function(e) {
    clearTimeout(searchTimeout);
    const query = e.target.value.trim();
    
    if (query.length < 2) {
        document.getElementById('endResults').classList.remove('show');
        return;
    }
    
    searchTimeout = setTimeout(() => search(query, 'end'), 500);
});

// 검색
async function search(query, type) {
    try {
        const res = await fetch('/api/search', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({query})
        });

        const data = await res.json();

        if (data.success && data.results.length > 0) {
            showResults(data.results, type);
        } else {
            hideResults(type);
        }
    } catch (err) {
        console.error('검색 오류:', err);
    }
}

// 결과 표시
function showResults(results, type) {
    const div = document.getElementById(type + 'Results');
    
    let html = '';
    results.forEach((r, i) => {
        html += `
            <div class="result-item" onclick="select(${i}, '${type}')">
                <div class="result-name">${r.name}</div>
                <div class="result-address">${r.address}</div>
            </div>
        `;
    });

    div.innerHTML = html;
    div.classList.add('show');
    window[type + 'Data'] = results;
}

// 결과 숨기기
function hideResults(type) {
    document.getElementById(type + 'Results').classList.remove('show');
}

// 선택
function select(i, type) {
    const result = window[type + 'Data'][i];
    
    if (type === 'start') {
        selectedStart = result;
        document.getElementById('start').value = result.name;
    } else {
        selectedEnd = result;
        document.getElementById('end').value = result.name;
    }
    
    hideResults(type);
}

// 지도 초기화
function initMap(startLat, startLng, endLat, endLng) {
    if (!map) {
        map = new Tmapv2.Map("map", {
            center: new Tmapv2.LatLng(startLat, startLng),
            width: "100%",
            height: "400px",
            zoom: 15
        });
    }

    // 기존 마커/폴리라인 제거
    if (map.clearOverlays) {
        map.clearOverlays();
    }

    // 출발지 마커
    new Tmapv2.Marker({
        position: new Tmapv2.LatLng(startLat, startLng),
        icon: "http://tmapapi.sktelecom.com/upload/tmap/marker/pin_r_m_s.png",
        map: map
    });

    // 도착지 마커
    new Tmapv2.Marker({
        position: new Tmapv2.LatLng(endLat, endLng),
        icon: "http://tmapapi.sktelecom.com/upload/tmap/marker/pin_r_m_e.png",
        map: map
    });

    document.getElementById('map').classList.add('show');
}

// 경로 그리기
function drawRoute(pathCoords) {
    if (!map || !pathCoords || pathCoords.length === 0) return;

    const tmapCoords = pathCoords.map(c => new Tmapv2.LatLng(c[0], c[1]));

    const polyline = new Tmapv2.Polyline({
        path: tmapCoords,
        strokeColor: "#FF6B6B",
        strokeWeight: 6,
        map: map
    });

    // 지도 범위 조정
    const bounds = new Tmapv2.LatLngBounds();
    tmapCoords.forEach(coord => bounds.extend(coord));
    map.fitBounds(bounds);

    // 경로 데이터 저장
    pathData = pathCoords;
}

// 경로 찾기
async function getDirections() {
    if (!selectedStart || !selectedEnd) {
        alert('출발지와 목적지를 검색해서 선택하세요');
        return;
    }

    const resultDiv = document.getElementById('result');
    resultDiv.innerHTML = '<div class="loading"><div class="spinner"></div>보행 경로 계산 중...</div>';

    document.getElementById('routeBtn').disabled = true;

    const start = `${selectedStart.lat},${selectedStart.lng}`;
    const end = `${selectedEnd.lat},${selectedEnd.lng}`;

    try {
        const res = await fetch('/api/directions', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({start, end})
        });

        const data = await res.json();

        if (res.ok && data.success) {
            routeData = data;
            currentStepIndex = 0;

            // 지도 초기화 및 경로 표시
            initMap(selectedStart.lat, selectedStart.lng, selectedEnd.lat, selectedEnd.lng);
            
            if (data.path_coords) {
                drawRoute(data.path_coords);
            }

            let html = `
                <div class="result-box">
                    <h3>🎯 보행 경로</h3>
                    
                    <div class="route-summary">
                        <div class="route-info">
                            <div class="route-info-item">
                                <div class="route-info-label">총 거리</div>
                                <div class="route-info-value">${data.distance_text}</div>
                            </div>
                            <div class="route-info-item">
                                <div class="route-info-label">예상 시간</div>
                                <div class="route-info-value">${data.duration_text}</div>
                            </div>
                        </div>
                    </div>
            `;

            if (data.steps && data.steps.length > 0) {
                html += '<div class="steps" id="stepsList"><h4 style="margin-bottom: 15px; color: #333;">턴바이턴 안내</h4>';
                
                data.steps.forEach((step, i) => {
                    html += `
                        <div class="step" id="step-${i}">
                            <div class="step-instruction">
                                <span class="step-number">${i + 1}</span>
                                ${step.instruction}
                            </div>
                            <div class="step-distance">${step.distance} · ${step.duration}</div>
                        </div>
                    `;
                });

                html += '</div>';
            }

            html += '</div>';
            resultDiv.innerHTML = html;

            // 실시간 안내 버튼 표시
            document.getElementById('navBtn').style.display = 'block';

            // 음성 재생
            if (data.audio) {
                new Audio('data:audio/mp3;base64,' + data.audio).play();
            }
        } else {
            resultDiv.innerHTML = `<div class="status error">❌ ${data.detail || '경로를 찾을 수 없습니다'}</div>`;
        }
    } catch (err) {
        resultDiv.innerHTML = `<div class="status error">❌ 오류: ${err.message}</div>`;
    } finally {
        document.getElementById('routeBtn').disabled = false;
    }
}

// 실시간 안내 시작
function startNavigation() {
    if (!routeData || !routeData.steps || routeData.steps.length === 0) {
        alert('먼저 경로를 찾으세요');
        return;
    }

    if (!navigator.geolocation) {
        alert('GPS를 지원하지 않는 브라우저입니다');
        return;
    }

    // GPS 추적 시작
    watchId = navigator.geolocation.watchPosition(
        updatePosition,
        handleGPSError,
        { enableHighAccuracy: true, maximumAge: 0, timeout: 5000 }
    );

    // UI 업데이트
    document.getElementById('gpsStatus').classList.add('show', 'active');
    document.getElementById('gpsText').textContent = '추적 중...';
    document.getElementById('currentStep').classList.add('show');
    document.getElementById('navBtn').style.display = 'none';
    document.getElementById('stopBtn').style.display = 'block';

    // 첫 안내
    updateCurrentStep();
}

// GPS 위치 업데이트
function updatePosition(position) {
    const lat = position.coords.latitude;
    const lng = position.coords.longitude;

    document.getElementById('currentLocation').textContent = `${lat.toFixed(6)}, ${lng.toFixed(6)}`;

    // 현재 위치 마커 업데이트
    if (marker) {
        marker.setPosition(new Tmapv2.LatLng(lat, lng));
    } else if (map) {
        marker = new Tmapv2.Marker({
            position: new Tmapv2.LatLng(lat, lng),
            icon: "http://tmapapi.sktelecom.com/upload/tmap/marker/pin_b_m_p.png",
            map: map
        });
    }

    // 다음 단계로 이동 판단
    checkNextStep(lat, lng);
}

// GPS 오류 처리
function handleGPSError(error) {
    console.error('GPS 오류:', error);
    document.getElementById('gpsText').textContent = 'GPS 오류 - ' + error.message;
}

// 현재 안내 업데이트
function updateCurrentStep() {
    if (currentStepIndex >= routeData.steps.length) {
        // 도착
        document.getElementById('currentInstruction').textContent = '🎉 목적지에 도착했습니다!';
        document.getElementById('distanceRemaining').textContent = '';
        speak('목적지에 도착했습니다');
        stopNavigation();
        return;
    }

    const step = routeData.steps[currentStepIndex];
    document.getElementById('currentInstruction').textContent = step.instruction;
    document.getElementById('distanceRemaining').textContent = `${step.distance} 남음`;

    // 현재 단계 강조
    document.querySelectorAll('.step').forEach(s => s.classList.remove('current'));
    const stepEl = document.getElementById(`step-${currentStepIndex}`);
    if (stepEl) {
        stepEl.classList.add('current');
        stepEl.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }

    // 음성 안내
    speak(step.instruction);
}

// 다음 단계 체크 (거리 기반)
function checkNextStep(lat, lng) {
    if (!pathData || pathData.length === 0) return;

    // 현재 위치에서 가장 가까운 경로 포인트 찾기
    let minDist = Infinity;
    let closestIndex = 0;

    pathData.forEach((point, i) => {
        const dist = getDistance(lat, lng, point[0], point[1]);
        if (dist < minDist) {
            minDist = dist;
            closestIndex = i;
        }
    });

    // 경로의 절반 이상 진행했으면 다음 단계로
    const progress = closestIndex / pathData.length;
    const stepProgress = (currentStepIndex + 1) / routeData.steps.length;

    if (progress > stepProgress && currentStepIndex < routeData.steps.length - 1) {
        currentStepIndex++;
        updateCurrentStep();
    }
}

// 두 지점 간 거리 계산 (Haversine)
function getDistance(lat1, lon1, lat2, lon2) {
    const R = 6371e3; // 지구 반지름 (미터)
    const φ1 = lat1 * Math.PI / 180;
    const φ2 = lat2 * Math.PI / 180;
    const Δφ = (lat2 - lat1) * Math.PI / 180;
    const Δλ = (lon2 - lon1) * Math.PI / 180;

    const a = Math.sin(Δφ / 2) * Math.sin(Δφ / 2) +
              Math.cos(φ1) * Math.cos(φ2) *
              Math.sin(Δλ / 2) * Math.sin(Δλ / 2);
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));

    return R * c; // 미터
}

// 음성 안내
function speak(text) {
    if ('speechSynthesis' in window) {
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.lang = 'ko-KR';
        utterance.rate = 0.9;
        speechSynthesis.speak(utterance);
    }
}

// 안내 중지
function stopNavigation() {
    if (watchId) {
        navigator.geolocation.clearWatch(watchId);
        watchId = null;
    }

    document.getElementById('gpsStatus').classList.remove('active');
    document.getElementById('gpsText').textContent = '중지됨';
    document.getElementById('currentStep').classList.remove('show');
    document.getElementById('navBtn').style.display = 'block';
    document.getElementById('stopBtn').style.display = 'none';

    if (marker && map) {
        marker.setMap(null);
        marker = null;
    }
}

// 외부 클릭 시 결과 숨기기
document.addEventListener('click', function(e) {
    if (!e.target.closest('.search-box')) {
        hideResults('start');
        hideResults('end');
    }
});
