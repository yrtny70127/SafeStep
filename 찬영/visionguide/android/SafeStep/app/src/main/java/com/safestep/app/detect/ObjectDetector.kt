package com.safestep.app.detect

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import android.graphics.RectF
import android.util.Log
import org.pytorch.IValue
import org.pytorch.LiteModuleLoader
import org.pytorch.Module
import org.pytorch.torchvision.TensorImageUtils
import kotlin.math.max
import kotlin.math.min

/**
 * 카메라 프레임을 받아서 [Detection] 리스트를 돌려주는 인터페이스.
 *
 * 구현체:
 *  - [NoOpDetector]   : 모델 파일이 없거나 로드 실패 시 사용. 항상 빈 리스트.
 *  - [PyTorchDetector]: assets/model.ptl + assets/labels.txt 로 동작.
 *
 * 사용 측은 항상 인터페이스에만 의존하고, [create] 가 상황에 맞는 구현을 골라준다.
 */
interface ObjectDetector {
    fun detect(bitmap: Bitmap, rotationDegrees: Int): List<Detection>
    fun close()

    companion object {
        private const val TAG = "ObjectDetector"
        const val MODEL_ASSET = "model.ptl"
        const val LABELS_ASSET = "labels.txt"

        fun create(context: Context): ObjectDetector {
            // 서버 모드 우선 — RemoteDetector 를 기본으로 사용합니다.
            // 서버 IP 변경 시: RemoteDetector.SERVER_URL 수정
            Log.i(TAG, "RemoteDetector 사용: ${RemoteDetector.SERVER_URL}")
            return RemoteDetector()
        }

        fun createLocal(context: Context): ObjectDetector {
            // 로컬 .ptl 모델이 있을 때 사용 (서버 없이 온디바이스 추론)
            val modelPath = AssetUtils.assetFilePath(context, MODEL_ASSET)
            val labels = AssetUtils.readAssetLines(context, LABELS_ASSET)
            if (modelPath == null) {
                Log.w(TAG, "model.ptl 없음 → NoOpDetector 사용")
                return NoOpDetector()
            }
            return runCatching {
                val module = LiteModuleLoader.load(modelPath)
                Log.i(TAG, "PyTorch 모델 로드 성공: $modelPath, labels=${labels.size}")
                PyTorchDetector(module, labels)
            }.getOrElse {
                Log.e(TAG, "모델 로드 실패 → NoOpDetector", it)
                NoOpDetector()
            }
        }
    }
}

/** 모델이 없을 때 쓰는 더미. 앱이 죽지 않게 해줌. */
class NoOpDetector : ObjectDetector {
    override fun detect(bitmap: Bitmap, rotationDegrees: Int): List<Detection> = emptyList()
    override fun close() {}
}

/**
 * PyTorch Mobile 디텍터.
 *
 * ⚠ 후처리 [parseRawOutput] 가 모델 출력 포맷에 따라 달라진다.
 * 현재는 YOLOv5/v8 계열을 가정한 골격이며, 팀 모델 사양 받으면 그 부분만 채우면 된다.
 * 모델 사양 문서: assets/MODEL_SPEC.md
 *
 * @param inputSize 모델이 기대하는 입력 한 변 크기 (정사각형 가정)
 * @param confThreshold 신뢰도 임계값 (이하 폐기)
 * @param iouThreshold NMS IoU 임계값
 */
class PyTorchDetector(
    private val module: Module,
    private val labels: List<String>,
    private val inputSize: Int = 640,
    private val confThreshold: Float = 0.4f,
    private val iouThreshold: Float = 0.5f,
) : ObjectDetector {

    override fun detect(bitmap: Bitmap, rotationDegrees: Int): List<Detection> {
        // 1) 회전 보정 + 정사각 리사이즈
        val rotated = if (rotationDegrees != 0) {
            val m = Matrix().apply { postRotate(rotationDegrees.toFloat()) }
            Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, m, true)
        } else bitmap

        val resized = Bitmap.createScaledBitmap(rotated, inputSize, inputSize, true)

        // 2) Tensor 변환 (ImageNet 평균/표준편차 사용. 팀 모델이 다르면 교체)
        val inputTensor = TensorImageUtils.bitmapToFloat32Tensor(
            resized,
            TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,
            TensorImageUtils.TORCHVISION_NORM_STD_RGB
        )

        // 3) 추론
        val output = module.forward(IValue.from(inputTensor))

        // 4) 후처리 — 모델 사양에 따라 채우기
        return parseRawOutput(output)
    }

    /**
     * TODO: 팀 모델 출력 포맷 확정되면 채워넣는다.
     * - YOLOv5/v8 export(.ptl) 라면 보통 [1, N, 5+num_classes] 텐서
     *   [cx, cy, w, h, obj_conf, cls_scores...]
     * - 이미 후처리 포함된 모델이면 [boxes, scores, labels] 튜플
     *
     * 후보 구현 예시는 MODEL_SPEC.md 참고.
     */
    private fun parseRawOutput(@Suppress("UNUSED_PARAMETER") output: IValue): List<Detection> {
        // 골격: 모델 사양 확정 전까지는 빈 리스트 반환.
        return emptyList()
    }

    override fun close() {
        runCatching { module.destroy() }
    }

    // 추후 후처리에서 사용할 NMS — 미리 만들어둠
    @Suppress("unused")
    private fun nonMaxSuppression(
        candidates: List<Detection>,
        iou: Float = iouThreshold,
    ): List<Detection> {
        val sorted = candidates.sortedByDescending { it.confidence }.toMutableList()
        val kept = mutableListOf<Detection>()
        while (sorted.isNotEmpty()) {
            val best = sorted.removeAt(0)
            kept += best
            sorted.removeAll { other ->
                other.label == best.label && iouOf(best.box, other.box) > iou
            }
        }
        return kept
    }

    private fun iouOf(a: RectF, b: RectF): Float {
        val left = max(a.left, b.left)
        val top = max(a.top, b.top)
        val right = min(a.right, b.right)
        val bottom = min(a.bottom, b.bottom)
        if (right <= left || bottom <= top) return 0f
        val inter = (right - left) * (bottom - top)
        val areaA = (a.right - a.left) * (a.bottom - a.top)
        val areaB = (b.right - b.left) * (b.bottom - b.top)
        return inter / (areaA + areaB - inter + 1e-6f)
    }
}
