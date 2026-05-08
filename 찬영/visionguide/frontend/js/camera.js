/**
 * 카메라 모듈
 * - getUserMedia로 후면 카메라 접근
 * - AI 분석용 JPEG를 2초 간격으로 서버에 전송
 */

export class CameraStream {
    constructor(opts) {
        this.videoEl     = opts.videoEl;
        this.canvasEl    = opts.canvasEl;
        this.intervalMs  = opts.intervalMs  ?? 2000;
        this.maxWidth    = opts.maxWidth    ?? 640;
        this.jpegQuality = opts.jpegQuality ?? 0.7;

        this.stream     = null;
        this.onFrame    = null;
        this._capturing = false;
        this._rafId     = null;
        this._lastSent  = 0;
    }

    async start() {
        if (!navigator.mediaDevices?.getUserMedia) {
            throw new Error('이 브라우저에서는 카메라를 사용할 수 없습니다.');
        }
        const constraints = {
            audio: false,
            video: { facingMode: { ideal: 'environment' }, width: { ideal: 1280 }, height: { ideal: 720 } },
        };
        try {
            this.stream = await navigator.mediaDevices.getUserMedia(constraints);
        } catch {
            this.stream = await navigator.mediaDevices.getUserMedia({ audio: false, video: true });
        }
        this.videoEl.srcObject = this.stream;
        await this.videoEl.play();
        const settings = this.stream.getVideoTracks()[0].getSettings();
        console.log('[camera] started', settings);
        return settings;
    }

    startCapturing(onFrame) {
        this.onFrame    = onFrame;
        this._capturing = true;
        this._lastSent  = 0;
        this._rafId = requestAnimationFrame(() => this._loop());
    }

    stopCapturing() {
        this._capturing = false;
        if (this._rafId) { cancelAnimationFrame(this._rafId); this._rafId = null; }
    }

    _loop() {
        if (!this._capturing) return;
        const now = performance.now();
        if (now - this._lastSent >= this.intervalMs) {
            const sent = this._captureOnce();
            if (sent) this._lastSent = now;
        }
        this._rafId = requestAnimationFrame(() => this._loop());
    }

    _captureOnce() {
        if (!this.videoEl.videoWidth) return false;
        const vw = this.videoEl.videoWidth, vh = this.videoEl.videoHeight;
        const scale = Math.min(1, this.maxWidth / vw);
        const w = Math.round(vw * scale), h = Math.round(vh * scale);
        this.canvasEl.width = w; this.canvasEl.height = h;
        this.canvasEl.getContext('2d').drawImage(this.videoEl, 0, 0, w, h);
        const dataUrl = this.canvasEl.toDataURL('image/jpeg', this.jpegQuality);
        if (this.onFrame) return this.onFrame(dataUrl);
        return false;
    }

    stop() {
        this.stopCapturing();
        if (this.stream) {
            for (const track of this.stream.getTracks()) track.stop();
            this.stream = null;
        }
        if (this.videoEl) this.videoEl.srcObject = null;
    }
}
