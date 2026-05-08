package com.safestep.app

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View
import com.safestep.app.detect.Detection

/**
 * PreviewView 위에 올라가는 바운딩박스 오버레이.
 * updateDetections() 로 최신 감지 결과를 넘기면 즉시 다시 그린다.
 */
class BoundingBoxOverlay @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null
) : View(context, attrs) {

    private var detections: List<Detection> = emptyList()

    // ── Paint 세트 ──────────────────────────────────────────────────────────
    private val boxPaint = Paint().apply {
        style       = Paint.Style.STROKE
        strokeWidth = 4f
        color       = Color.parseColor("#F97316")  // 오렌지
        isAntiAlias = true
    }

    private val fillPaint = Paint().apply {
        style = Paint.Style.FILL
        color = Color.parseColor("#99F97316")      // 반투명 오렌지 배경 (라벨용)
    }

    private val textPaint = Paint().apply {
        color       = Color.WHITE
        textSize    = 36f
        isAntiAlias = true
        isFakeBoldText = true
    }

    private val dangerBoxPaint = Paint().apply {
        style       = Paint.Style.STROKE
        strokeWidth = 6f
        color       = Color.parseColor("#FF0000")  // 빨강 (매우 가까운 객체)
        isAntiAlias = true
    }

    /** 탐지 결과 갱신 — 백그라운드 스레드에서 호출해도 OK */
    fun updateDetections(list: List<Detection>) {
        detections = list
        postInvalidate()   // 어느 스레드에서든 안전하게 재드로우 요청
    }

    fun clear() {
        detections = emptyList()
        postInvalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        if (detections.isEmpty()) return

        val w = width.toFloat()
        val h = height.toFloat()

        for (det in detections) {
            val left   = det.box.left   * w
            val top    = det.box.top    * h
            val right  = det.box.right  * w
            val bottom = det.box.bottom * h
            val rect   = RectF(left, top, right, bottom)

            // 면적 0.4 이상이면 빨간 박스(매우 가까움), 아니면 오렌지
            val paint = if (det.area() >= 0.40f) dangerBoxPaint else boxPaint
            canvas.drawRoundRect(rect, 8f, 8f, paint)

            // 라벨 배경 + 텍스트
            val label    = "${det.label}  ${(det.confidence * 100).toInt()}%"
            val textW    = textPaint.measureText(label)
            val textH    = textPaint.textSize
            val labelTop = if (top - textH - 10f < 0) bottom else top - textH - 10f

            canvas.drawRoundRect(
                RectF(left, labelTop, left + textW + 16f, labelTop + textH + 8f),
                4f, 4f, fillPaint
            )
            canvas.drawText(label, left + 8f, labelTop + textH, textPaint)
        }
    }
}
