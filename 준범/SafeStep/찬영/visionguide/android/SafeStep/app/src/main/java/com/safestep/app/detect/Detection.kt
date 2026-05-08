package com.safestep.app.detect

import android.graphics.RectF

/**
 * 객체 감지 결과 1개를 표현하는 표준 데이터 형식.
 *
 * @property label 클래스 라벨 (예: "person", "car"). 한국어 라벨은 UI 단에서 매핑.
 * @property confidence 신뢰도 0.0 ~ 1.0
 * @property box 정규화된 좌표 (0..1, 좌상단 (0,0)). 화면 비율로 곱해서 픽셀 좌표 얻음.
 *           area() 로 박스 크기 비율을 바로 얻을 수 있음.
 */
data class Detection(
    val label: String,
    val confidence: Float,
    val box: RectF,
    val direction: String = "정면",   // 서버가 보내는 방향 (왼쪽/정면/오른쪽)
    val depthM: Float? = null,         // 서버가 보내는 거리(m), 없으면 null
) {
    /** 박스가 화면에서 차지하는 비율 (0..1) — 거리 추정에 사용. */
    fun area(): Float = box.width().coerceAtLeast(0f) * box.height().coerceAtLeast(0f)

    /** 박스 가로 중심이 화면 어디에 있는지 (0..1). 0.5 가 정면. */
    fun centerX(): Float = (box.left + box.right) / 2f
}
