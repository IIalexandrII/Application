package ru.nstu.navigator_arcore;

import android.app.Activity;
import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Rect;
import android.os.SystemClock;
import android.util.Log;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.GpuDelegate;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import ru.nstu.navigator_arcore.tools.BoundingBox;
import ru.nstu.navigator_arcore.tools.ImageTools;


/**
 * Универсальная обертка над TFLite моделью для детекции.
 *
 * Я делал так, чтобы оно:
 * 1) само понимало формат выхода (готовые детекции или raw YOLO),
 * 2) работало в фоне и не душило GL/UI поток,
 * 3) отдавало тайминги (prep/run/decode), чтобы можно было смотреть производительность.
 */
public class Model {

    private static final String TAG = "ModelTFLite";

    // ====================== ТЮНИНГ ======================
    // Эти значения подбираются под устройство — можно смело крутить и смотреть по таймингам.
    public static final int THREADS_GPU = 2;     // обычно 1-2 норм, 4 иногда даже хуже
    public static final int THREADS_CPU = 4;     // зависит от big.LITTLE, но 4 чаще ок
    public static final boolean GPU_SUSTAINED_SPEED = true; // sustained = стабильнее fps, fast_single = быстрее, но рывками
    public static final long MIN_DETECT_INTERVAL_MS = 250;  // чтобы не пытаться детектить каждый кадр (и не греть телефон)

    // Пороги детекции/постобработки
    public static float CONF = 0.25f;
    public static float NMS_IOU = 0.45f;
    public static int MAX_DETS = 50;

    // ====================================================

    // Лейблы для отображения в оверлее
    public final String[] Classes;

    private final Interpreter interpreter;
    private final GpuDelegate gpuDelegate; // null, если GPU не завелся и упали на CPU
    private final Activity activity;

    // Вход у модели фиксированный (под мой экспорт): 640x640
    private static final int IN_W = 640;
    private static final int IN_H = 640;

    // Я делаю center-crop в квадрат и потом ресайз в 640x640 через Canvas
    private final Rect srcRect = new Rect();
    private final Rect dstRect = new Rect(0, 0, IN_W, IN_H);

    private final Bitmap inputBitmap;
    private final Canvas inputCanvas;

    // Буферы под преобразование Bitmap -> float32 RGB
    private final int[] inputPixels = new int[IN_W * IN_H];
    private final ByteBuffer inputBuffer =
            ByteBuffer.allocateDirect(1 * IN_W * IN_H * 3 * 4).order(ByteOrder.nativeOrder());

    // ----- output mode detection -----
    // У разных экспортов/конвертаций выход может быть разный, поэтому я определяю формат по shape.
    private enum OutputMode {
        DET_300x6,      // [1,300,6] (x1,y1,x2,y2,score,cls)
        YOLO_RAW_3D,    // [1,C,N] или [1,N,C] (bbox + class scores)
        UNKNOWN
    }
    private final OutputMode outMode;
    private final int[] outShape0;
    private final int outTensors;

    // Буферы под выход. Я заранее аллоцирую, чтобы не плодить мусор на каждом кадре.
    private final float[][][] out3; // для 3D выхода
    private final float[][] out2;   // если вдруг модель дает 2D (на всякий)

    // async
    // Один поток мне хватает: важно не параллелить детекции, а не блокировать рендер.
    private final ExecutorService exec = Executors.newSingleThreadExecutor();
    private volatile List<BoundingBox> lastBoxes = new ArrayList<>();
    private volatile boolean busy = false;
    private volatile long lastDetectMs = 0;

    // Тайминги последнего прогона, чтобы потом можно было выводить в UI/debug.
    private volatile float lastPrepMs = 0f;
    private volatile float lastRunMs = 0f;
    private volatile float lastDecMs = 0f;
    private volatile float lastTotalMs = 0f;
    private volatile long lastFinishUptimeMs = 0;

    public float getLastTotalMs() { return lastTotalMs; }
    public float getLastRunMs()   { return lastRunMs; }
    public float getLastPrepMs()  { return lastPrepMs; }
    public float getLastDecMs()   { return lastDecMs; }
    public long getLastFinishUptimeMs() { return lastFinishUptimeMs; }


    public static void setCONF(float conf){CONF = conf;}
    public static void setNMS_IOU(float nms_iou){NMS_IOU = nms_iou;}
    public static void setMAX_DETS(int max_dets){MAX_DETS = max_dets;}
    public static float getCONF(){return CONF;}
    public static float getNMS_IOU(){return NMS_IOU;}
    public static int getMAX_DETS(){return MAX_DETS;}


    public Model(String modelAssetName,
                       String labelsAssetName,
                       Context context,
                       boolean useGpu) throws IOException {

        // Мне нужен Activity, чтобы корректно грузить ассеты и логать инфу по девайсу.
        if (!(context instanceof Activity)) throw new IOException("Context is not Activity");
        this.activity = (Activity) context;

        Interpreter.Options opt = new Interpreter.Options();
        GpuDelegate tmpGpu = null;

        // В идеале работаем на GPU: быстрее и меньше грузим CPU.
        if (useGpu) {
            opt.setNumThreads(THREADS_GPU);
            opt.setUseXNNPACK(false); // если GPU — XNNPACK обычно не нужен

            try {
                // Настройки GPU делегата. Я разрешаю fp16 (precisionLossAllowed),
                // потому что для детекции это обычно ок и дает ускорение.
                GpuDelegate.Options gpuOpt = new GpuDelegate.Options();
                gpuOpt.setPrecisionLossAllowed(true);
                gpuOpt.setInferencePreference(
                        GPU_SUSTAINED_SPEED
                                ? GpuDelegate.Options.INFERENCE_PREFERENCE_SUSTAINED_SPEED
                                : GpuDelegate.Options.INFERENCE_PREFERENCE_FAST_SINGLE_ANSWER
                );

                tmpGpu = new GpuDelegate(gpuOpt);
                opt.addDelegate(tmpGpu);

                Log.i(TAG, "GPU delegate enabled (" +
                        (GPU_SUSTAINED_SPEED ? "SUSTAINED_SPEED" : "FAST_SINGLE_ANSWER") +
                        ", FP16 allowed)");
            } catch (Throwable t) {
                // Если GPU делегат не завелся — не падаю, просто работаю на CPU.
                Log.w(TAG, "GPU delegate not available, fallback to CPU", t);
                tmpGpu = null;
                opt.setNumThreads(THREADS_CPU);
                opt.setUseXNNPACK(true);
            }
        } else {
            // Чисто CPU режим. XNNPACK обычно ускоряет на ARM.
            opt.setNumThreads(THREADS_CPU);
            opt.setUseXNNPACK(true);
        }

        gpuDelegate = tmpGpu;
        interpreter = new Interpreter(loadModelFile(activity, modelAssetName), opt);

        // Логи — чтобы сразу понимать что именно загрузилось и какие у модели тензоры.
        Log.i(TAG, "useGpu=" + useGpu + " gpuDelegate=" + (gpuDelegate != null));
        Log.i(TAG, "threads(gpu)=" + THREADS_GPU + " threads(cpu)=" + THREADS_CPU);
        Log.i(TAG, "inputs=" + interpreter.getInputTensorCount() + " outputs=" + interpreter.getOutputTensorCount());

        Log.i(TAG, "inTensor=" + Arrays.toString(interpreter.getInputTensor(0).shape())
                + " type=" + interpreter.getInputTensor(0).dataType()
                + " name=" + interpreter.getInputTensor(0).name());

        outTensors = interpreter.getOutputTensorCount();
        outShape0 = interpreter.getOutputTensor(0).shape();

        for (int i = 0; i < outTensors; i++) {
            Log.i(TAG, "out[" + i + "]=" + Arrays.toString(interpreter.getOutputTensor(i).shape())
                    + " type=" + interpreter.getOutputTensor(i).dataType()
                    + " name=" + interpreter.getOutputTensor(i).name());
        }

        Log.i(TAG, "ABI=" + android.os.Build.SUPPORTED_ABIS[0] +
                " device=" + android.os.Build.MANUFACTURER + " " + android.os.Build.MODEL);

        // Гружу лейблы из assets (по строке на класс)
        Classes = loadLabels(activity, labelsAssetName);

        // Промежуточный bitmap под ресайз/кроп (чтобы не аллоцировать каждый раз)
        inputBitmap = Bitmap.createBitmap(IN_W, IN_H, Bitmap.Config.ARGB_8888);
        inputCanvas = new Canvas(inputBitmap);

        // Понимаю, какой формат выхода у модели, и под него выделяю буфер.
        OutputMode mode = detectOutputMode(outShape0);
        outMode = mode;

        float[][][] tmpOut3 = null;
        float[][] tmpOut2 = null;

        // В TFLite удобнее заранее создать массив нужной формы и потом в него писать вывод.
        if (outShape0.length == 3) {
            int d1 = outShape0[1] > 0 ? outShape0[1] : 1;
            int d2 = outShape0[2] > 0 ? outShape0[2] : 1;
            tmpOut3 = new float[1][d1][d2];
        } else if (outShape0.length == 2) {
            int d1 = outShape0[1] > 0 ? outShape0[1] : 1;
            tmpOut2 = new float[1][d1];
        } else {
            // Если shape вообще непонятный — пробую как самый частый кейс [1,300,6]
            tmpOut3 = new float[1][300][6];
        }

        out3 = tmpOut3;
        out2 = tmpOut2;

        Log.i(TAG, "Detected outMode=" + outMode + " outShape0=" + Arrays.toString(outShape0));

        // Прогрев: первый инференс на Android часто с лагом (JIT/кеши/аллоцирование).
        try {
            for (int i = 0; i < 3; i++) runOnce();
            Log.i(TAG, "Warmup done");
        } catch (Throwable t) {
            Log.w(TAG, "Warmup failed", t);
        }
    }

    /**
     * По форме первого выхода пытаюсь понять, что это:
     * - детекции 300x6
     * - или raw YOLO с большим N
     */
    private OutputMode detectOutputMode(int[] shape) {
        if (shape == null) return OutputMode.UNKNOWN;
        if (shape.length == 3 && shape[0] == 1) {
            int a = shape[1];
            int b = shape[2];
            // [1,300,6]
            if ((a == 300 && b == 6) || (a == 6 && b == 300)) return OutputMode.DET_300x6;

            // raw YOLO: [1,C,N] или [1,N,C], где N обычно большой (тысячи якорей/точек)
            if ((a >= 5 && b >= 1000) || (b >= 5 && a >= 1000)) return OutputMode.YOLO_RAW_3D;
        }
        return OutputMode.UNKNOWN;
    }

    /**
     * Освобождаю ресурсы. Важно закрывать delegate и interpreter, иначе утечки/краши на некоторых девайсах.
     */
    public void close() {
        try { exec.shutdownNow(); } catch (Throwable ignored) {}
        try { interpreter.close(); } catch (Throwable ignored) {}
        try { if (gpuDelegate != null) gpuDelegate.close(); } catch (Throwable ignored) {}
    }

    /**
     * Асинхронная детекция.
     * Я специально не блокирую рендер: если модель занята или еще рано по интервалу —
     * отдаю lastBoxes (последний готовый результат).
     */
    public List<BoundingBox> analyzeImageAsync(Bitmap image,
                                               int origCamW,
                                               int origCamH,
                                               int rotationDeg) {

        if (image == null) return lastBoxes;
        if (exec.isShutdown() || exec.isTerminated()) return lastBoxes;

        long now = SystemClock.uptimeMillis();
        // если уже что-то считается — не спамлю задачами
        if (busy) return lastBoxes;
        // лимит по частоте, чтобы не упираться в 100% загрузку
        if (now - lastDetectMs < MIN_DETECT_INTERVAL_MS) return lastBoxes;

        lastDetectMs = now;
        busy = true;

        // Важно: если внешний код переиспользует bitmap, лучше делать copy(),
        // иначе можем словить гонку (тут оставил как есть, потому что зависит от пайплайна).
        final Bitmap frame = image;

        exec.execute(() -> {
            try {
                List<BoundingBox> res = analyzeImageSync(frame, origCamW, origCamH, rotationDeg);
                lastBoxes = (res != null) ? res : new ArrayList<>();
            } catch (Throwable t) {
                Log.e(TAG, "analyzeImageAsync failed", t);
            } finally {
                busy = false;
            }
        });

        return lastBoxes;
    }

    // ---------------- main sync path (runs in background) ----------------
    /**
     * Основной синхронный путь (но вызываю я его из фонового потока).
     * Тут:
     * 1) crop+resize
     * 2) packing в float32 буфер
     * 3) inference
     * 4) decode (det300 или raw yolo)
     * 5) маппинг координат обратно в координаты камеры + undoRotate
     */
    private List<BoundingBox> analyzeImageSync(Bitmap image,
                                               int origCamW,
                                               int origCamH,
                                               int rotationDeg) {

        if (image == null) return new ArrayList<>();

        int camW = image.getWidth();
        int camH = image.getHeight();

        // center crop до квадрата, чтобы не ломать пропорции (и соответствовать тому, как модель обучалась)
        int cropSize = Math.min(camW, camH);
        int cropX = (camW - cropSize) / 2;
        int cropY = (camH - cropSize) / 2;

        long tPrep0 = System.nanoTime();

        // Рисую кусок кадра в 640x640 bitmap (быстро и без лишних матриц руками)
        srcRect.set(cropX, cropY, cropX + cropSize, cropY + cropSize);
        inputCanvas.drawBitmap(image, srcRect, dstRect, null);

        // Bitmap -> float32 RGB (0..1)
        fillInputBufferFromBitmap(inputBitmap, inputPixels, inputBuffer);

        long tPrep1 = System.nanoTime();

        // inference
        long tRun0 = System.nanoTime();
        runOnce();
        long tRun1 = System.nanoTime();

        long tDec0 = System.nanoTime();

        // decode по заранее определенному режиму
        List<BoundingBox> boxes;

        if (outMode == OutputMode.DET_300x6) {
            // поддерживаю оба варианта раскладки, которые иногда получаются после конвертации
            boxes = decodeDetections300x6(out3, cropSize, cropSize);
        } else if (outMode == OutputMode.YOLO_RAW_3D) {
            // numClasses беру из labels, если они есть, иначе дефолт 80
            int numClasses = (Classes != null) ? Classes.length : 80;
            boxes = decodeYoloRawAuto(out3, cropSize, cropSize, numClasses);
        } else {
            // на всякий — пробую трактовать как [1,300,6]
            boxes = decodeDetections300x6(out3, cropSize, cropSize);
        }

        long tDec1 = System.nanoTime();

        // считаю тайминги для отладки производительности
        float prepMs = (tPrep1 - tPrep0) / 1_000_000f;
        float runMs  = (tRun1  - tRun0)  / 1_000_000f;
        float decMs  = (tDec1  - tDec0)  / 1_000_000f;

        lastPrepMs = prepMs;
        lastRunMs = runMs;
        lastDecMs = decMs;
        lastTotalMs = prepMs + runMs + decMs;
        lastFinishUptimeMs = SystemClock.uptimeMillis();

        Log.i("YOLO_TIMES",
                "mode=" + outMode +
                        " prep=" + prepMs +
                        "ms run=" + runMs +
                        "ms dec=" + decMs +
                        "ms total=" + lastTotalMs + "ms"
        );

        // Маппинг боксов: из координат кропа обратно в координаты исходного кадра
        for (BoundingBox b : boxes) {
            if (b == null || b.rect == null) continue;

            // сначала просто смещаю на cropX/cropY
            b.rect.offset(cropX, cropY);

            // потом учитываю поворот, чтобы rect совпал с тем, как камера отдает изображение в UI
            b.rect = ImageTools.undoRotate(
                    b.rect,
                    origCamW,
                    origCamH,
                    ImageTools.getTotalRotation(rotationDeg)
            );

            // на всякий подрезаю, чтобы не улететь за границы
            if (b.rect.left < 0) b.rect.left = 0;
            if (b.rect.top < 0) b.rect.top = 0;
            if (b.rect.right > camW) b.rect.right = camW;
            if (b.rect.bottom > camH) b.rect.bottom = camH;
        }

        return boxes;
    }

    /**
     * Один прогон inference. Выход пишется в заранее выделенный буфер.
     */
    private void runOnce() {
        if (out3 != null) {
            interpreter.run(inputBuffer, out3);
        } else if (out2 != null) {
            interpreter.run(inputBuffer, out2);
        } else {
            // супер запасной вариант
            interpreter.run(inputBuffer, new float[1][300][6]);
        }
    }

    // ===================== Decoder A: [1,300,6] or [1,6,300] =====================

    /**
     * Декодер для готовых детекций формата:
     * [x1,y1,x2,y2,score,cls]
     *
     * Иногда после конвертации размерности меняются местами, поэтому поддерживаю:
     * - [300][6]
     * - [6][300]
     */
    private static List<BoundingBox> decodeDetections300x6(float[][][] out, int imageW, int imageH) {
        ArrayList<BoundingBox> res = new ArrayList<>();
        if (out == null || out.length == 0 || out[0] == null) return res;

        int d1 = out[0].length;
        int d2 = (d1 > 0 && out[0][0] != null) ? out[0][0].length : 0;

        // вариант [300][6]
        if (d1 == 300 && d2 >= 6) {
            for (int i = 0; i < 300; i++) {
                float[] d = out[0][i];
                addDet300x6(res, d, imageW, imageH);
            }
            return res;
        }

        // вариант [6][300]
        if (d1 == 6 && d2 == 300) {
            for (int i = 0; i < 300; i++) {
                float x1 = out[0][0][i];
                float y1 = out[0][1][i];
                float x2 = out[0][2][i];
                float y2 = out[0][3][i];
                float score = out[0][4][i];
                float clsF = out[0][5][i];
                float[] d = new float[]{x1, y1, x2, y2, score, clsF};
                addDet300x6(res, d, imageW, imageH);
            }
            return res;
        }

        // если не попали ни в один кейс — пробую интерпретировать строки как детекции
        for (int i = 0; i < d1; i++) {
            float[] d = out[0][i];
            if (d == null || d.length < 6) continue;
            addDet300x6(res, d, imageW, imageH);
        }
        return res;
    }

    /**
     * Привожу одну детекцию к RectF + фильтрую по conf.
     * Нормированные координаты (0..1) тоже поддерживаю.
     */
    private static void addDet300x6(List<BoundingBox> res, float[] d, int imageW, int imageH) {
        if (d == null || d.length < 6) return;

        float x1 = d[0], y1 = d[1], x2 = d[2], y2 = d[3];
        float score = d[4];
        int cls = (int) d[5];

        // порог по уверенности
        if (score < CONF) return;

        // некоторые модели отдают координаты в 0..1, некоторые в пикселях — определяю по диапазону
        boolean norm = (Math.max(Math.max(x1, y1), Math.max(x2, y2)) <= 1.5f);
        if (norm) {
            x1 *= imageW; x2 *= imageW;
            y1 *= imageH; y2 *= imageH;
        }

        // на всякий нормализую порядок углов
        float left = Math.min(x1, x2);
        float right = Math.max(x1, x2);
        float top = Math.min(y1, y2);
        float bottom = Math.max(y1, y2);

        // подрезаю в границы изображения
        if (left < 0) left = 0;
        if (top < 0) top = 0;
        if (right > imageW) right = imageW;
        if (bottom > imageH) bottom = imageH;

        // отсекаю микробоксы (шум)
        if ((right - left) < 2f || (bottom - top) < 2f) return;

        res.add(new BoundingBox(new android.graphics.RectF(left, top, right, bottom), score, cls));
    }

    // ===================== Decoder B: raw YOLO ([1,C,N] or [1,N,C]) =====================

    /**
     * Декодер raw YOLO:
     * - сам выбирает раскладку [C,N] или [N,C]
     * - предполагаю, что первые 4 — bbox (cx,cy,w,h)
     * - дальше class scores
     * - после этого делаю NMS по каждому классу
     */
    private static List<BoundingBox> decodeYoloRawAuto(float[][][] out, int imageW, int imageH, int numClasses) {
        if (out == null || out.length == 0 || out[0] == null) return new ArrayList<>();
        int A = out[0].length;
        int B = (A > 0 && out[0][0] != null) ? out[0][0].length : 0;
        if (A <= 0 || B <= 0) return new ArrayList<>();

        // Пытаюсь понять layout: где каналы, а где количество предсказаний.
        boolean layoutCN;
        if (A >= 5 && A <= 512 && B >= 1000) layoutCN = true;       // похоже на [C,N]
        else if (B >= 5 && B <= 512 && A >= 1000) layoutCN = false; // похоже на [N,C]
        else {
            // совсем эвристика: обычно C меньше N
            layoutCN = A < B;
        }

        if (layoutCN) {
            // out[0][c][i]
            int C = A;
            int N = B;
            int classes = Math.min(numClasses, C - 4);
            return decodeYoloRaw_CN(out, imageW, imageH, classes, C, N);
        } else {
            // out[0][i][c]
            int N = A;
            int C = B;
            int classes = Math.min(numClasses, C - 4);
            return decodeYoloRaw_NC(out, imageW, imageH, classes, C, N);
        }
    }

    /**
     * Внутренний класс под детекцию для NMS.
     * Я разделяю Det и BoundingBox, чтобы NMS был проще и не трогать внешний тип.
     */
    private static class Det {
        android.graphics.RectF r;
        float s;
        int c;
        Det(android.graphics.RectF r, float s, int c) { this.r = r; this.s = s; this.c = c; }
    }

    /**
     * Декодирование формата [C,N]: out[0][channel][index]
     */
    private static List<BoundingBox> decodeYoloRaw_CN(float[][][] out, int imageW, int imageH, int classes, int C, int N) {
        ArrayList<Det> dets = new ArrayList<>();

        for (int i = 0; i < N; i++) {
            float cx = out[0][0][i];
            float cy = out[0][1][i];
            float w  = out[0][2][i];
            float h  = out[0][3][i];

            // беру лучший класс (argmax)
            int bestCls = -1;
            float bestScore = -1f;
            for (int c = 0; c < classes; c++) {
                float sc = out[0][4 + c][i];
                if (sc > bestScore) { bestScore = sc; bestCls = c; }
            }
            if (bestScore < CONF) continue;

            // координаты могут быть нормированные — опять же угадываю по диапазону
            boolean norm = (Math.max(Math.max(cx, cy), Math.max(w, h)) <= 2.0f);
            if (norm) { cx *= imageW; cy *= imageH; w *= imageW; h *= imageH; }

            // (cx,cy,w,h) -> (l,t,r,b)
            float left = cx - w / 2f;
            float top = cy - h / 2f;
            float right = cx + w / 2f;
            float bottom = cy + h / 2f;

            // clamp
            if (left < 0) left = 0;
            if (top < 0) top = 0;
            if (right > imageW) right = imageW;
            if (bottom > imageH) bottom = imageH;

            if ((right - left) < 2f || (bottom - top) < 2f) continue;
            dets.add(new Det(new android.graphics.RectF(left, top, right, bottom), bestScore, bestCls));
        }

        // NMS по классам, чтобы не выкидывать разные классы друг другом
        ArrayList<Det> kept = nmsPerClass(dets, NMS_IOU, MAX_DETS);
        ArrayList<BoundingBox> res = new ArrayList<>(kept.size());
        for (Det d : kept) res.add(new BoundingBox(d.r, d.s, d.c));
        return res;
    }

    /**
     * Декодирование формата [N,C]: out[0][index][channel]
     */
    private static List<BoundingBox> decodeYoloRaw_NC(float[][][] out, int imageW, int imageH, int classes, int C, int N) {
        ArrayList<Det> dets = new ArrayList<>();

        for (int i = 0; i < N; i++) {
            float cx = out[0][i][0];
            float cy = out[0][i][1];
            float w  = out[0][i][2];
            float h  = out[0][i][3];

            int bestCls = -1;
            float bestScore = -1f;
            for (int c = 0; c < classes; c++) {
                float sc = out[0][i][4 + c];
                if (sc > bestScore) { bestScore = sc; bestCls = c; }
            }
            if (bestScore < CONF) continue;

            boolean norm = (Math.max(Math.max(cx, cy), Math.max(w, h)) <= 2.0f);
            if (norm) { cx *= imageW; cy *= imageH; w *= imageW; h *= imageH; }

            float left = cx - w / 2f;
            float top = cy - h / 2f;
            float right = cx + w / 2f;
            float bottom = cy + h / 2f;

            if (left < 0) left = 0;
            if (top < 0) top = 0;
            if (right > imageW) right = imageW;
            if (bottom > imageH) bottom = imageH;

            if ((right - left) < 2f || (bottom - top) < 2f) continue;
            dets.add(new Det(new android.graphics.RectF(left, top, right, bottom), bestScore, bestCls));
        }

        ArrayList<Det> kept = nmsPerClass(dets, NMS_IOU, MAX_DETS);
        ArrayList<BoundingBox> res = new ArrayList<>(kept.size());
        for (Det d : kept) res.add(new BoundingBox(d.r, d.s, d.c));
        return res;
    }

    /**
     * NMS отдельно по каждому классу.
     * Сначала группирую детекции по classId, сортирую по score и выбрасываю пересекающиеся.
     */
    private static ArrayList<Det> nmsPerClass(List<Det> dets, float iouThr, int maxOut) {
        HashMap<Integer, ArrayList<Det>> byCls = new HashMap<>();
        for (Det d : dets) {
            ArrayList<Det> list = byCls.get(d.c);
            if (list == null) {
                list = new ArrayList<>();
                byCls.put(d.c, list);
            }
            list.add(d);
        }

        ArrayList<Det> out = new ArrayList<>();

        for (ArrayList<Det> list : byCls.values()) {
            // сортирую по уверенности по убыванию
            list.sort((a, b) -> Float.compare(b.s, a.s));

            boolean[] removed = new boolean[list.size()];
            for (int i = 0; i < list.size(); i++) {
                if (removed[i]) continue;
                Det a = list.get(i);
                out.add(a);
                if (out.size() >= maxOut) return out;

                // выкидываю всё, что сильно пересекается с текущим лучшим
                for (int j = i + 1; j < list.size(); j++) {
                    if (removed[j]) continue;
                    Det b = list.get(j);
                    if (iou(a.r, b.r) >= iouThr) removed[j] = true;
                }
            }
        }

        // на выходе сортирую все классы вместе (чтобы удобнее рисовать топ)
        out.sort((a, b) -> Float.compare(b.s, a.s));
        if (out.size() > maxOut) return new ArrayList<>(out.subList(0, maxOut));
        return out;
    }

    /**
     * Intersection-over-Union для двух прямоугольников.
     */
    private static float iou(android.graphics.RectF a, android.graphics.RectF b) {
        float interLeft = Math.max(a.left, b.left);
        float interTop = Math.max(a.top, b.top);
        float interRight = Math.min(a.right, b.right);
        float interBottom = Math.min(a.bottom, b.bottom);

        float iw = interRight - interLeft;
        float ih = interBottom - interTop;
        if (iw <= 0 || ih <= 0) return 0f;

        float inter = iw * ih;
        float areaA = (a.right - a.left) * (a.bottom - a.top);
        float areaB = (b.right - b.left) * (b.bottom - b.top);
        float union = areaA + areaB - inter;
        return union <= 0 ? 0f : (inter / union);
    }

    // ---------------- input packing ----------------
    /**
     * Упаковка bitmap в inputBuffer в формате float32 RGB (0..1).
     * Порядок: r,g,b подряд на каждый пиксель.
     */
    private static void fillInputBufferFromBitmap(Bitmap bmp, int[] pixels, ByteBuffer buf) {
        bmp.getPixels(pixels, 0, IN_W, 0, 0, IN_W, IN_H);

        buf.rewind();
        for (int p : pixels) {
            float r = ((p >> 16) & 0xFF) / 255f;
            float g = ((p >> 8) & 0xFF) / 255f;
            float b = (p & 0xFF) / 255f;
            buf.putFloat(r);
            buf.putFloat(g);
            buf.putFloat(b);
        }
        buf.rewind();
    }

    // ---------------- assets helpers ----------------
    private static String[] loadLabels(Context ctx, String path) {
        ArrayList<String> list = new ArrayList<>();
        try {
            InputStream is;
            File file = new File(path);
            if (file.exists()) {
                is = new FileInputStream(file);
            } else {
                is = ctx.getAssets().open(path);
            }

            try (BufferedReader br = new BufferedReader(new InputStreamReader(is))) {
                String line;
                while ((line = br.readLine()) != null) {
                    line = line.trim();
                    if (!line.isEmpty()) list.add(line);
                }
            }
            return list.toArray(new String[0]);
        } catch (Exception e) {
            Log.e("Model", "Failed to load labels: " + path, e);
            return new String[]{"error"};
        }
    }

    /**
     * Гружу .tflite из assets через mmap (быстрее и меньше копий в памяти).
     */
    private static MappedByteBuffer loadModelFile(Context context, String path) throws IOException {
        File file = new File(path);
        if (file.exists()) {
            // Загрузка из внутреннего хранилища
            try (FileInputStream fis = new FileInputStream(file);
                 FileChannel fileChannel = fis.getChannel()) {
                return fileChannel.map(FileChannel.MapMode.READ_ONLY, 0, file.length());
            }
        } else {
            // Загрузка из Assets
            AssetFileDescriptor afd = context.getAssets().openFd(path);
            try (FileInputStream fis = new FileInputStream(afd.getFileDescriptor());
                 FileChannel fileChannel = fis.getChannel()) {
                return fileChannel.map(FileChannel.MapMode.READ_ONLY, afd.getStartOffset(), afd.getDeclaredLength());
            }
        }
    }
}