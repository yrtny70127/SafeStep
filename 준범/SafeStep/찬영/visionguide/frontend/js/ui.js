/**
 * UI 컨트롤러
 * - 화면 전환: 홈 ↔ 길찾기 ↔ 카메라
 * - 길찾기: T-map 검색 + 경로 + 미니맵
 * - 카메라: WebSocket + AI 분석 결과
 * - TTS: 길안내 + AI 위험 감지 통합
 */

import { CameraStream } from '/static/js/camera.js';
import { CameraSocket } from '/static/js/websocket.js';

// ============================================
// DOM 참조
// ============================================
const screens = {
    home:   document.getElementById('home-screen'),
    nav:    document.getElementById('nav-screen'),
    camera: document.getElementById('camera-screen'),
};

const homeBtns = {
    cameraMode: document.getElementById('btn-camera-mode'),
    navMode:    document.getElementById('btn-nav-mode'),
};

const camEls = {
    video:        document.getElementById('camera-video'),
    canvas:       document.getElementById('capture-canvas'),
    backBtn:      document.getElementById('btn-back'),
    toggleBtn:    document.getElementById('btn-toggle'),
    captureLabel: document.getElementById('capture-label'),
    wsDot:        document.getElementById('ws-dot'),
    wsText:       document.getElementById('ws-text'),
    counter:      document.querySelector('#frame-counter .counter-num'),
    message:      document.querySelector('#latest-message .msg-text'),
    lastSize:     document.getElementById('last-size'),
    miniMap:      document.getElementById('mini-map'),
    navBar:       document.getElementById('nav-bar'),
    navBarText:   document.getElementById('nav-bar-text'),
    navBarDist:   document.getElementById('nav-bar-dist'),
    navBarIcon:   document.getElementById('nav-bar-icon'),
};

const navEls = {
    backBtn:      document.getElementById('btn-nav-back'),
    destInput:    document.getElementById('nav-dest-input'),
    voiceBtn:     document.getElementById('nav-voice-btn'),
    results:      document.getElementById('nav-results'),
    routeCard:    document.getElementById('nav-route-card'),
    routeDist:    document.getElementById('nav-route-dist'),
    routeTime:    document.getElementById('nav-route-time'),
    steps:        document.getElementById('nav-steps'),
    mapPreview:   document.getElementById('nav-map-preview'),
    findBtn:      document.getElementById('btn-find-route'),
    startBtn:     document.getElementById('btn-nav-start'),
};

const serverStatus = {
    dot:  document.querySelector('#server-status .conn-dot'),
    text: document.querySelector('#server-status .conn-text'),
};

const overlay = {
    el:       document.getElementById('permission-overlay'),
    text:     document.getElementById('permission-text'),
    retryBtn: document.getElementById('btn-retry-permission'),
};

// ============================================
// 상태
// ============================================
let camera     = null;
let socket     = null;
let isStreaming = false;
let frameCount = 0;

// 길찾기 상태
let navState = {
    active:       false,    // 길안내 진행 중 여부
    dest:         null,     // { name, lat, lng }
    routeData:    null,     // 서버 응답 경로 데이터
    currentStep:  0,        // 현재 안내 단계 인덱스
    watchId:      null,     // GPS watchPosition ID
    previewMap:   null,     // 길찾기 화면 T-map 지도
    miniMapObj:   null,     // 카메라 화면 미니맵
    miniMapMarker:null,     // 현재 위치 마커
    tmapKey:      null,     // T-map 앱키
    tmapLoaded:   false,    // SDK 로드 여부
};

// TTS 상태
let ttsQueue     = [];     // { text, priority }
let ttsSpeaking  = false;
const TTS_PRIO = { NAV: 3, DANGER: 2, INFO: 1 };

// 마지막 AI 음성 시간 + 메시지 (반복 방지)
let lastAiSpeak = 0;
let lastAiMsg = '';

// ============================================
// 화면 전환
// ============================================
function showScreen(name) {
    for (const [key, el] of Object.entries(screens)) {
        el.classList.toggle('active', key === name);
    }
}

// ============================================
// 서버 상태 확인
// ============================================
async function checkServerHealth() {
    try {
        const res = await fetch('/health', { cache: 'no-store' });
        if (res.ok) {
            serverStatus.dot.dataset.state = 'ok';
            serverStatus.text.textContent = '서버 연결 정상';
        } else throw new Error();
    } catch {
        serverStatus.dot.dataset.state = 'err';
        serverStatus.text.textContent = '서버 응답 없음';
    }
}

// ============================================
// TTS (우선순위 큐)
// ============================================
function speak(text, priority = TTS_PRIO.INFO) {
    if (!('speechSynthesis' in window) || !text) return;

    // 낮은 우선순위는 큐에 쌓지 않고 무시 (길안내 중 잡음 제거)
    if (navState.active && priority < TTS_PRIO.DANGER) return;

    // 현재 말하는 것보다 높은 우선순위면 즉시 교체
    if (ttsSpeaking && priority >= TTS_PRIO.NAV) {
        speechSynthesis.cancel();
        ttsSpeaking = false;
    }

    ttsQueue.push({ text, priority });
    ttsQueue.sort((a, b) => b.priority - a.priority);
    _speakNext();
}

function _speakNext() {
    if (ttsSpeaking || ttsQueue.length === 0) return;
    const { text } = ttsQueue.shift();
    const utt = new SpeechSynthesisUtterance(text);
    utt.lang = 'ko-KR';
    utt.rate = 1.0;
    utt.onend = () => { ttsSpeaking = false; _speakNext(); };
    utt.onerror = () => { ttsSpeaking = false; _speakNext(); };
    ttsSpeaking = true;
    speechSynthesis.speak(utt);
}

// ============================================
// T-map SDK 동적 로드
// ============================================
function loadTmapSDK(key) {
    return new Promise((resolve, reject) => {
        if (navState.tmapLoaded) { resolve(); return; }
        const script = document.createElement('script');
        script.src = `https://apis.openapi.sk.com/tmap/jsv2?version=1&appKey=${key}`;
        script.onload = () => { navState.tmapLoaded = true; resolve(); };
        script.onerror = reject;
        document.head.appendChild(script);
    });
}

async function initTmap() {
    if (navState.tmapKey) return;
    try {
        const res = await fetch('/api/nav/config');
        const { tmap_key } = await res.json();
        navState.tmapKey = tmap_key;
        await loadTmapSDK(tmap_key);
    } catch (e) {
        console.error('[tmap] SDK 로드 실패', e);
    }
}

// ============================================
// 길찾기 화면
// ============================================
async function enterNavScreen() {
    showScreen('nav');
    navEls.destInput.value = '';
    navEls.results.innerHTML = '';
    navEls.routeCard.classList.add('hidden');
    navEls.startBtn.classList.add('hidden');
    navEls.mapPreview.classList.add('hidden');
    navEls.findBtn.disabled = true;
    navState.dest = null;
    navState.routeData = null;

    // T-map SDK 미리 로드
    await initTmap();
}

// 목적지 검색
let searchTimer = null;
navEls.destInput.addEventListener('input', () => {
    const q = navEls.destInput.value.trim();
    navEls.findBtn.disabled = true;
    navEls.routeCard.classList.add('hidden');
    navEls.startBtn.classList.add('hidden');
    navState.dest = null;
    navState.routeData = null;

    clearTimeout(searchTimer);
    if (q.length < 2) { navEls.results.innerHTML = ''; return; }
    searchTimer = setTimeout(() => doSearch(q), 400);
});

async function doSearch(query) {
    try {
        const res = await fetch('/api/nav/search', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query }),
        });
        const data = await res.json();
        renderResults(data.results || []);
    } catch {
        navEls.results.innerHTML = '<div style="padding:12px;color:var(--fg-mute)">검색 실패</div>';
    }
}

function renderResults(results) {
    if (!results.length) {
        navEls.results.innerHTML = '<div style="padding:12px;color:var(--fg-mute)">결과 없음</div>';
        return;
    }
    navEls.results.innerHTML = results.map((r, i) => `
        <div class="nav-result-item" data-i="${i}">
            <div class="nav-result-name">${r.name}</div>
            <div class="nav-result-addr">${r.address}</div>
        </div>
    `).join('');
    navEls.results.querySelectorAll('.nav-result-item').forEach(el => {
        el.addEventListener('click', () => selectDest(results[+el.dataset.i]));
    });
}

function selectDest(r) {
    navState.dest = r;
    navEls.destInput.value = r.name;
    navEls.results.innerHTML = '';
    navEls.findBtn.disabled = false;
}

// 경로 찾기
navEls.findBtn.addEventListener('click', async () => {
    if (!navState.dest) return;

    navEls.findBtn.disabled = true;
    navEls.findBtn.textContent = '경로 계산 중…';

    // GPS로 현재 위치 가져오기
    let startLat, startLng;
    try {
        const pos = await new Promise((res, rej) =>
            navigator.geolocation.getCurrentPosition(res, rej, { enableHighAccuracy: true, timeout: 8000 })
        );
        startLat = pos.coords.latitude;
        startLng = pos.coords.longitude;
    } catch {
        alert('GPS를 사용할 수 없습니다. 위치 권한을 허용해주세요.');
        navEls.findBtn.disabled = false;
        navEls.findBtn.textContent = '경로 찾기';
        return;
    }

    try {
        const res = await fetch('/api/nav/directions', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                start_lat: startLat, start_lng: startLng,
                end_lat: navState.dest.lat, end_lng: navState.dest.lng,
            }),
        });
        if (!res.ok) throw new Error(await res.text());
        navState.routeData = await res.json();
        navState.startPos = { lat: startLat, lng: startLng };
        renderRoute(navState.routeData);
        renderPreviewMap(startLat, startLng);
    } catch (e) {
        alert('경로 계산 실패: ' + e.message);
    } finally {
        navEls.findBtn.disabled = false;
        navEls.findBtn.textContent = '경로 찾기';
    }
});

function renderRoute(data) {
    navEls.routeDist.textContent = data.distance_text;
    navEls.routeTime.textContent = data.duration_text;
    navEls.steps.innerHTML = data.steps.map((s, i) => `
        <div class="nav-step">
            <span class="nav-step-num">${i + 1}</span>
            <div class="nav-step-texts">
                <div class="nav-step-text">${s.instruction}</div>
                ${s.distance ? `<div class="nav-step-dist">${s.distance}m</div>` : ''}
            </div>
        </div>
    `).join('');
    navEls.routeCard.classList.remove('hidden');
    navEls.startBtn.classList.remove('hidden');

    speak(`${navState.dest.name}까지 ${data.distance_text}, ${data.duration_text}입니다.`, TTS_PRIO.INFO);
}

function renderPreviewMap(startLat, startLng) {
    if (!navState.tmapLoaded || !window.Tmapv2) return;
    navEls.mapPreview.classList.remove('hidden');

    try {
        if (navState.previewMap) navState.previewMap.destroy();
        navState.previewMap = new Tmapv2.Map('nav-map-preview', {
            center: new Tmapv2.LatLng(startLat, startLng),
            width: '100%', height: '200px', zoom: 15,
        });

        const data = navState.routeData;
        if (data?.path_coords?.length) {
            const coords = data.path_coords.map(c => new Tmapv2.LatLng(c[0], c[1]));
            new Tmapv2.Polyline({ path: coords, strokeColor: '#fff200', strokeWeight: 5, map: navState.previewMap });
            const bounds = new Tmapv2.LatLngBounds();
            coords.forEach(c => bounds.extend(c));
            navState.previewMap.fitBounds(bounds);
        }
    } catch (e) {
        console.warn('[tmap] 미리보기 지도 오류', e);
    }
}

// 음성 입력
navEls.voiceBtn.addEventListener('click', () => {
    if (!('webkitSpeechRecognition' in window || 'SpeechRecognition' in window)) {
        alert('이 브라우저는 음성 인식을 지원하지 않습니다.');
        return;
    }
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    const recognition = new SR();
    recognition.lang = 'ko-KR';
    recognition.interimResults = false;
    recognition.maxAlternatives = 1;

    navEls.voiceBtn.classList.add('listening');
    recognition.start();

    recognition.onresult = (e) => {
        const text = e.results[0][0].transcript;
        navEls.destInput.value = text;
        navEls.destInput.dispatchEvent(new Event('input'));
    };
    recognition.onend = () => navEls.voiceBtn.classList.remove('listening');
    recognition.onerror = () => navEls.voiceBtn.classList.remove('listening');
});

// 경로 안내 시작
navEls.startBtn.addEventListener('click', () => {
    navState.active = true;
    navState.currentStep = 0;
    enterCameraMode(true); // 길찾기 모드로 카메라 진입
});

// ============================================
// 카메라 모드 진입
// ============================================
async function enterCameraMode(withNav = false) {
    showScreen('camera');

    camera = new CameraStream({
        videoEl: camEls.video,
        canvasEl: camEls.canvas,
        intervalMs: 2000,
        maxWidth: 640,
        jpegQuality: 0.7,
    });

    try {
        await camera.start();
    } catch (err) {
        showPermissionOverlay(err);
        return;
    }

    setupWebSocket();

    if (withNav && navState.routeData) {
        startNavigation();
    }
}

// ============================================
// GPS 길안내 (카메라 화면)
// ============================================
function startNavigation() {
    camEls.navBar.classList.remove('hidden');
    camEls.miniMap.classList.remove('hidden');
    updateNavBar();
    speak(navState.routeData.steps[0]?.instruction || '길안내를 시작합니다', TTS_PRIO.NAV);
    // DOM이 렌더링된 뒤 지도 초기화
    setTimeout(initMiniMap, 300);

    navState.watchId = navigator.geolocation.watchPosition(
        onGpsUpdate,
        (e) => console.warn('[gps]', e),
        { enableHighAccuracy: true, maximumAge: 0, timeout: 10000 }
    );
}

function initMiniMap() {
    if (!navState.tmapLoaded || !window.Tmapv2 || !navState.startPos) return;
    try {
        const { lat, lng } = navState.startPos;
        navState.miniMapObj = new Tmapv2.Map('mini-map', {
            center: new Tmapv2.LatLng(lat, lng),
            width: '140px', height: '140px', zoom: 16,
        });
        // 경로 폴리라인
        const coords = navState.routeData.path_coords.map(c => new Tmapv2.LatLng(c[0], c[1]));
        if (coords.length) {
            new Tmapv2.Polyline({ path: coords, strokeColor: '#fff200', strokeWeight: 4, map: navState.miniMapObj });
        }
    } catch (e) {
        console.warn('[minimap]', e);
    }
}

function onGpsUpdate(pos) {
    const lat = pos.coords.latitude;
    const lng = pos.coords.longitude;

    // 미니맵 현재 위치 마커 업데이트
    if (navState.miniMapObj && window.Tmapv2) {
        try {
            const ll = new Tmapv2.LatLng(lat, lng);
            navState.miniMapObj.setCenter(ll);
            if (navState.miniMapMarker) {
                navState.miniMapMarker.setPosition(ll);
            } else {
                navState.miniMapMarker = new Tmapv2.Marker({
                    position: ll,
                    map: navState.miniMapObj,
                });
            }
        } catch {}
    }

    // 다음 단계 체크
    checkNextStep(lat, lng);
}

function checkNextStep(lat, lng) {
    const data = navState.routeData;
    if (!data?.path_coords?.length) return;

    const total = data.steps.length;
    if (navState.currentStep >= total) return;

    // 경로 진행도로 현재 단계 추정
    let minDist = Infinity, closest = 0;
    data.path_coords.forEach((c, i) => {
        const d = haversine(lat, lng, c[0], c[1]);
        if (d < minDist) { minDist = d; closest = i; }
    });

    const progress = closest / data.path_coords.length;
    const stepProgress = (navState.currentStep + 1) / total;

    if (progress > stepProgress && navState.currentStep < total - 1) {
        navState.currentStep++;
        updateNavBar();
        speak(data.steps[navState.currentStep].instruction, TTS_PRIO.NAV);
    }

    // 도착 판정 (목적지 30m 이내)
    if (haversine(lat, lng, navState.dest.lat, navState.dest.lng) < 30) {
        speak('목적지에 도착했습니다', TTS_PRIO.NAV);
        stopNavigation();
    }
}

function updateNavBar() {
    const step = navState.routeData?.steps[navState.currentStep];
    if (!step) return;
    camEls.navBarText.textContent = step.instruction;
    camEls.navBarDist.textContent = step.distance ? `${step.distance}m` : '';
}

function stopNavigation() {
    if (navState.watchId != null) {
        navigator.geolocation.clearWatch(navState.watchId);
        navState.watchId = null;
    }
    navState.active = false;
    camEls.navBar.classList.add('hidden');
    camEls.miniMap.classList.add('hidden');
}

function haversine(lat1, lon1, lat2, lon2) {
    const R = 6371e3;
    const f1 = lat1 * Math.PI / 180, f2 = lat2 * Math.PI / 180;
    const df = (lat2 - lat1) * Math.PI / 180;
    const dl = (lon2 - lon1) * Math.PI / 180;
    const a = Math.sin(df/2)**2 + Math.cos(f1)*Math.cos(f2)*Math.sin(dl/2)**2;
    return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
}

// ============================================
// WebSocket
// ============================================
function setupWebSocket() {
    socket = new CameraSocket({
        onOpen:    () => setWsState('ok', '연결됨'),
        onClose:   () => setWsState('warn', '재연결 중'),
        onError:   () => setWsState('err', '연결 오류'),
        onMessage: (data) => handleServerMessage(data),
    });
    socket.connect();
}

function handleServerMessage(data) {
    if (data.type === 'ack') {
        camEls.message.textContent = data.message || '서버에 연결되었습니다';
    } else if (data.type === 'result' || data.type === 'frame') {
        frameCount = data.frame_id ?? frameCount + 1;
        camEls.counter.textContent = String(frameCount);

        const msg = data.message || '';
        camEls.message.textContent = msg;
        if (data.image_size_kb != null) camEls.lastSize.textContent = `${data.image_size_kb} KB`;

        // AI 음성: 5초마다 최대 1회 + 같은 메시지 반복 안 함 + 이상없음 생략
        const now = Date.now();
        const isDanger = msg.includes('위험') || msg.includes('접근') || msg.includes('차도') || msg.includes('빨간불');
        const interval = isDanger ? 3000 : 6000; // 위험은 3초, 일반은 6초
        if (msg && msg !== '이상 없음' && msg !== lastAiMsg && now - lastAiSpeak > interval) {
            const prio = isDanger ? TTS_PRIO.DANGER : TTS_PRIO.INFO;
            speak(msg, prio);
            lastAiSpeak = now;
            lastAiMsg = msg;
        }
    } else if (data.type === 'error') {
        camEls.message.textContent = `⚠️ ${data.message}`;
    }
}

function setWsState(state, text) {
    camEls.wsDot.dataset.state = state;
    camEls.wsText.textContent = text;
}

// ============================================
// 시작/정지 토글
// ============================================
function toggleStreaming() {
    if (!camera) return;
    if (isStreaming) {
        camera.stopCapturing();
        isStreaming = false;
        camEls.toggleBtn.classList.remove('active');
        camEls.toggleBtn.setAttribute('aria-label', '전송 시작');
        camEls.captureLabel.textContent = 'START';
        camEls.message.textContent = '전송이 중지되었습니다';
    } else {
        if (!socket?.isOpen()) {
            camEls.message.textContent = '서버 연결을 기다리는 중입니다…';
            return;
        }
        camera.startCapturing((dataUrl) => socket.sendFrame(dataUrl));
        isStreaming = true;
        camEls.toggleBtn.classList.add('active');
        camEls.toggleBtn.setAttribute('aria-label', '전송 중지');
        camEls.captureLabel.textContent = 'STOP';
        camEls.message.textContent = '전송 중… 분석 결과를 기다리는 중';
    }
}

// ============================================
// 카메라 화면 종료 (뒤로가기)
// ============================================
function exitCameraMode() {
    stopNavigation();
    if (camera) { camera.stop(); camera = null; }
    if (socket) { socket.close(); socket = null; }
    isStreaming = false;
    frameCount = 0;
    camEls.toggleBtn.classList.remove('active');
    camEls.captureLabel.textContent = 'START';
    camEls.counter.textContent = '0';
    camEls.message.textContent = '시작 버튼을 눌러주세요';
    camEls.lastSize.textContent = '-';
    setWsState('idle', '대기 중');
    speechSynthesis.cancel();

    // 길찾기에서 왔으면 길찾기 화면으로, 아니면 홈으로
    if (navState.routeData) {
        showScreen('nav');
    } else {
        showScreen('home');
        checkServerHealth();
    }
}

// ============================================
// 권한 오버레이
// ============================================
function showPermissionOverlay(err) {
    let msg = '브라우저에서 카메라 사용을 허용해주세요.';
    if (err?.name === 'NotAllowedError') {
        msg = '카메라 권한이 거부되었습니다.<br>브라우저 설정에서 권한을 허용해주세요.';
    } else if (err?.name === 'NotFoundError') {
        msg = '카메라를 찾을 수 없습니다.';
    } else if (window.location.protocol !== 'https:' && window.location.hostname !== 'localhost') {
        msg = 'HTTPS 환경이 아니어서 카메라 접근이 차단되었습니다.<br>ngrok URL로 접속해주세요.';
    }
    overlay.text.innerHTML = msg;
    overlay.el.hidden = false;
}

function hidePermissionOverlay() { overlay.el.hidden = true; }

// ============================================
// 이벤트 바인딩
// ============================================
homeBtns.cameraMode.addEventListener('click', () => enterCameraMode(false));
homeBtns.navMode.addEventListener('click', enterNavScreen);
navEls.backBtn.addEventListener('click', () => { showScreen('home'); checkServerHealth(); });
camEls.backBtn.addEventListener('click', exitCameraMode);
camEls.toggleBtn.addEventListener('click', toggleStreaming);
overlay.retryBtn.addEventListener('click', async () => { hidePermissionOverlay(); await enterCameraMode(false); });

// ============================================
// 초기화
// ============================================
checkServerHealth();
