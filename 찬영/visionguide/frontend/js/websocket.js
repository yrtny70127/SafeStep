/**
 * WebSocket 클라이언트
 *
 * 서버와의 실시간 통신을 담당한다.
 * - ws/wss 프로토콜 자동 선택 (HTTPS면 wss)
 * - 자동 재연결 (지수 백오프)
 * - 이벤트 기반 콜백 (onOpen, onMessage, onClose, onError)
 */

export class CameraSocket {
    /**
     * @param {object} opts
     * @param {string} [opts.path='/ws/camera']  WebSocket 엔드포인트 경로
     * @param {function} [opts.onOpen]
     * @param {function} [opts.onMessage]   (data) => void
     * @param {function} [opts.onClose]
     * @param {function} [opts.onError]
     */
    constructor(opts = {}) {
        this.path = opts.path || '/ws/camera';
        this.onOpen = opts.onOpen || (() => {});
        this.onMessage = opts.onMessage || (() => {});
        this.onClose = opts.onClose || (() => {});
        this.onError = opts.onError || (() => {});

        this.ws = null;
        this.shouldReconnect = false;
        this.reconnectAttempts = 0;
        this.maxReconnectDelay = 8000;   // 최대 8초
    }

    /** 서버 URL을 현재 페이지의 호스트 기반으로 자동 결정 */
    _buildUrl() {
        const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        return `${proto}//${window.location.host}${this.path}`;
    }

    connect() {
        this.shouldReconnect = true;
        this._open();
    }

    _open() {
        const url = this._buildUrl();
        console.log('[ws] connecting →', url);

        try {
            this.ws = new WebSocket(url);
        } catch (e) {
            console.error('[ws] WebSocket 생성 실패', e);
            this._scheduleReconnect();
            return;
        }

        this.ws.onopen = () => {
            console.log('[ws] open');
            this.reconnectAttempts = 0;
            this.onOpen();
        };

        this.ws.onmessage = (ev) => {
            let data;
            try {
                data = JSON.parse(ev.data);
            } catch {
                console.warn('[ws] non-JSON message', ev.data);
                return;
            }
            this.onMessage(data);
        };

        this.ws.onerror = (ev) => {
            console.warn('[ws] error', ev);
            this.onError(ev);
        };

        this.ws.onclose = (ev) => {
            console.log('[ws] close', ev.code, ev.reason);
            this.onClose(ev);

            if (this.shouldReconnect) {
                this._scheduleReconnect();
            }
        };
    }

    _scheduleReconnect() {
        this.reconnectAttempts += 1;
        const delay = Math.min(
            500 * Math.pow(2, this.reconnectAttempts),
            this.maxReconnectDelay
        );
        console.log(`[ws] reconnect in ${delay}ms (attempt #${this.reconnectAttempts})`);
        setTimeout(() => {
            if (this.shouldReconnect) this._open();
        }, delay);
    }

    /** 프레임 전송 (data URL string). 버퍼가 찼으면 스킵해서 지연 누적 방지. */
    sendFrame(dataUrl) {
        if (!this.isOpen()) return false;
        if (this.ws.bufferedAmount > 131072) return false; // 128KB 초과 시 스킵

        const payload = JSON.stringify({
            type: 'frame',
            image: dataUrl,
            timestamp: Date.now(),
        });

        this.ws.send(payload);
        return true;
    }

    /** JSON 객체 전송 (WebRTC 시그널링 등) */
    sendJson(obj) {
        if (!this.isOpen()) return false;
        this.ws.send(JSON.stringify(obj));
        return true;
    }

    isOpen() {
        return this.ws && this.ws.readyState === WebSocket.OPEN;
    }

    close() {
        this.shouldReconnect = false;
        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }
    }
}
