package ru.nstu.navigator_arcore.modelTools;

import android.graphics.RectF;
import android.util.Log;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

import ru.nstu.navigator_arcore.tools.BoundingBox;

public class PostProcessor {

    // Минимальная уверенность детекции. Всё, что ниже — считаю шумом
    private static float CONFIDENCE_THRESHOLD = 0.5f;

    // Порог для NMS: если боксы сильно перекрываются — оставляю самый уверенный
    private static float IOU_THRESHOLD = 0.7f;

    // Ограничиваю число финальных детекций, чтобы не захламлять экран
    private static int MAX_DETECTIONS = 50;

    /**
     * Разбираю выход YOLOv8 и превращаю его в список BoundingBox.
     *
     * Модель может вернуть тензор в двух форматах:
     *  - [1][84][8400]
     *  - [1][8400][84]
     * поэтому формат определяю автоматически по размерам.
     *
     * В каждом предсказании:
     *  - cx, cy — центр объекта
     *  - w, h   — ширина и высота
     *  - дальше — вероятности классов
     *
     * Координаты бывают либо 0..1, либо в пикселях модели —
     * это тоже определяю на лету.
     *
     * Внутри метода:
     *  - отбрасываю слабые предсказания
     *  - перевожу координаты в пиксели изображения
     *  - убираю дубли через NMS
     *
     * На выходе получаю готовые боксы для отрисовки поверх камеры.
     */

    public static void setConfidenceThreshold(float value) {CONFIDENCE_THRESHOLD = value;}
    public static void setIouThreshold(float value) {IOU_THRESHOLD = value;}
    public static void setMaxDetections(int value) {MAX_DETECTIONS = value;}

    public static float getConfidenceThreshold() {return CONFIDENCE_THRESHOLD;}
    public static float getIouThreshold() {return IOU_THRESHOLD;}
    public static int getMaxDetections() {return MAX_DETECTIONS;}

    public static List<BoundingBox> process(float[][][] out, int inputSize, int imageW, int imageH) {
        if (out == null || out.length == 0 || out[0] == null) return new ArrayList<>();

        final int dim1 = out[0].length;
        final int dim2 = out[0][0].length;

        // Определяю layout выхода модели: [features][preds] или [preds][features]
        // Обычно features ~ 84, preds ~ 8400
        final boolean channelsFirst;
        final int rows, cols;

        if (dim1 <= 256 && dim2 > dim1) {
            channelsFirst = true;  // [84][8400]
            rows = dim1;
            cols = dim2;
        } else if (dim2 <= 256 && dim1 > dim2) {
            channelsFirst = false; // [8400][84]
            rows = dim2;
            cols = dim1;
        } else {
            // Если формат неочевиден — беру самый частый вариант
            channelsFirst = true;
            rows = dim1;
            cols = dim2;
        }

        // Должно быть минимум 4 координаты + хотя бы один класс
        if (rows < 5) return new ArrayList<>();
        final int numClasses = rows - 4;

        // Понимаю, в чем координаты: нормализованные или в пикселях модели
        // Смотрю максимум на небольшой выборке
        float maxCoord = 0f;
        int probe = Math.min(cols, 200);
        for (int i = 0; i < probe; i++) {
            float cx = get(out, channelsFirst, 0, i);
            float cy = get(out, channelsFirst, 1, i);
            float w  = get(out, channelsFirst, 2, i);
            float h  = get(out, channelsFirst, 3, i);
            maxCoord = Math.max(maxCoord, Math.max(Math.max(cx, cy), Math.max(w, h)));
        }

        // Если значения явно больше 1 — значит это пиксели, а не 0..1
        final boolean coordsArePixels = maxCoord > 2.5f;

        List<BoundingBox> candidates = new ArrayList<>();

        // Прохожусь по всем предсказаниям модели
        for (int i = 0; i < cols; i++) {
            float cx = get(out, channelsFirst, 0, i);
            float cy = get(out, channelsFirst, 1, i);
            float w  = get(out, channelsFirst, 2, i);
            float h  = get(out, channelsFirst, 3, i);

            // Ищу самый вероятный класс
            int bestClass = -1;
            float bestScore = 0f;
            for (int c = 0; c < numClasses; c++) {
                float s = get(out, channelsFirst, 4 + c, i);
                if (s > bestScore) {
                    bestScore = s;
                    bestClass = c;
                }
            }

            // Слабые предсказания сразу отбрасываю
            if (bestScore < CONFIDENCE_THRESHOLD) continue;

            // Если координаты были в пикселях модели — нормализую в 0..1
            if (coordsArePixels) {
                cx /= (float) inputSize;
                cy /= (float) inputSize;
                w  /= (float) inputSize;
                h  /= (float) inputSize;
            }

            // Перевожу (cx,cy,w,h) в обычный прямоугольник
            float x1 = (cx - w / 2f) * imageW;
            float y1 = (cy - h / 2f) * imageH;
            float x2 = (cx + w / 2f) * imageW;
            float y2 = (cy + h / 2f) * imageH;

            // На всякий случай нормализую стороны
            float left = Math.min(x1, x2);
            float right = Math.max(x1, x2);
            float top = Math.min(y1, y2);
            float bottom = Math.max(y1, y2);

            // Обрезаю по границам изображения
            left = clamp(left, 0f, imageW);
            right = clamp(right, 0f, imageW);
            top = clamp(top, 0f, imageH);
            bottom = clamp(bottom, 0f, imageH);

            // Слишком маленькие боксы — это почти всегда шум
            if ((right - left) < 2f || (bottom - top) < 2f) continue;

            candidates.add(new BoundingBox(new RectF(left, top, right, bottom), bestScore, bestClass));
        }

        // Убираю дубли (NMS отдельно для каждого класса)
        return nmsClassAware(candidates, IOU_THRESHOLD, MAX_DETECTIONS);
    }

    // Хелпер, чтобы не писать везде if(channelsFirst)
    private static float get(float[][][] out, boolean channelsFirst, int row, int col) {
        return channelsFirst ? out[0][row][col] : out[0][col][row];
    }

    // NMS по классам: боксы разных классов не подавляют друг друга
    private static List<BoundingBox> nmsClassAware(List<BoundingBox> boxes, float iouThreshold, int limit) {
        if (boxes.isEmpty()) return new ArrayList<>();

        // Сортирую по уверенности (сильные идут первыми)
        Collections.sort(boxes, new Comparator<BoundingBox>() {
            @Override public int compare(BoundingBox a, BoundingBox b) {
                return Float.compare(b.score, a.score);
            }
        });

        List<BoundingBox> selected = new ArrayList<>();
        for (BoundingBox a : boxes) {
            boolean keep = true;

            // Если уже есть бокс того же класса с большим IoU — выкидываю
            for (BoundingBox s : selected) {
                if (a.clazz == s.clazz && iou(a.rect, s.rect) > iouThreshold) {
                    keep = false;
                    break;
                }
            }

            if (keep) {
                selected.add(a);
                if (selected.size() >= limit) break;
            }
        }
        return selected;
    }

    // IoU = насколько сильно два бокса пересекаются (0..1)
    private static float iou(RectF a, RectF b) {
        float interLeft = Math.max(a.left, b.left);
        float interTop = Math.max(a.top, b.top);
        float interRight = Math.min(a.right, b.right);
        float interBottom = Math.min(a.bottom, b.bottom);

        float iw = Math.max(0f, interRight - interLeft);
        float ih = Math.max(0f, interBottom - interTop);
        float inter = iw * ih;

        float areaA = Math.max(0f, a.right - a.left) * Math.max(0f, a.bottom - a.top);
        float areaB = Math.max(0f, b.right - b.left) * Math.max(0f, b.bottom - b.top);

        float union = areaA + areaB - inter;
        if (union <= 0f) return 0f;

        // Небольшая защита от деления на ноль
        return inter / (union + 1e-6f);
    }

    // Обрезаю значение по диапазону
    private static float clamp(float v, float lo, float hi) {
        return Math.max(lo, Math.min(hi, v));
    }

    // ---  TEST

    public static List<BoundingBox> processFlatWithNMS(float[] data, long[] shape, int imageW, int imageH) {
        int numDetections = (int) shape[1]; // 300
        List<BoundingBox> candidates = new ArrayList<>();

        for (int i = 0; i < numDetections; i++) {
            int base = i * 6;
            float x1 = data[base];
            float y1 = data[base + 1];
            float x2 = data[base + 2];
            float y2 = data[base + 3];
            float score = data[base + 4];
            int cls = (int) data[base + 5];

            if (score < CONFIDENCE_THRESHOLD) continue;

            // Привожу координаты к пикселям view (если нужно)
            x1 = clamp(x1 * imageW, 0, imageW);
            y1 = clamp(y1 * imageH, 0, imageH);
            x2 = clamp(x2 * imageW, 0, imageW);
            y2 = clamp(y2 * imageH, 0, imageH);

            candidates.add(new BoundingBox(new RectF(x1, y1, x2, y2), score, cls));
        }

        // NMS класс-осознанная
        return nmsClassAware(candidates, IOU_THRESHOLD, MAX_DETECTIONS);
    }

    public static List<BoundingBox> processFlat(float[] data, long[] shape, int inputSize, int imageW, int imageH) {
        if (shape.length != 3) return new ArrayList<>();

        int B = (int) shape[0];
        int D1 = (int) shape[1];
        int D2 = (int) shape[2];

        final boolean channelsFirst;
        final int rows, cols;

        if (D1 <= 256 && D2 > D1) {
            channelsFirst = true;   // [1,84,8400]
            rows = D1;
            cols = D2;
        } else {
            channelsFirst = false;  // [1,8400,84]
            rows = D2;
            cols = D1;
        }

        final int numClasses = rows - 4;
        if (numClasses <= 0) return new ArrayList<>();

        float maxCoord = 0f;
        int probe = Math.min(cols, 200);
        for (int i = 0; i < probe; i++) {
            float cx = get(data, shape, channelsFirst, 0, i);
            float cy = get(data, shape, channelsFirst, 1, i);
            float w  = get(data, shape, channelsFirst, 2, i);
            float h  = get(data, shape, channelsFirst, 3, i);
            maxCoord = Math.max(maxCoord, Math.max(Math.max(cx, cy), Math.max(w, h)));
        }

        final boolean coordsArePixels = maxCoord > 2.5f;

        List<BoundingBox> candidates = new ArrayList<>();

        for (int i = 0; i < cols; i++) {
            float cx = get(data, shape, channelsFirst, 0, i);
            float cy = get(data, shape, channelsFirst, 1, i);
            float w  = get(data, shape, channelsFirst, 2, i);
            float h  = get(data, shape, channelsFirst, 3, i);

            if (coordsArePixels) {
                cx /= (float) inputSize;
                cy /= (float) inputSize;
                w  /= (float) inputSize;
                h  /= (float) inputSize;
            }
            int bestClass = -1;
            float bestScore = 0f;

            for (int c = 0; c < numClasses; c++) {
                float s = get(data, shape, channelsFirst, 4 + c, i);
                if (s > bestScore) {
                    bestScore = s;
                    bestClass = c;
                }
            }

            if (bestScore < CONFIDENCE_THRESHOLD) continue;

            float x1 = (cx - w / 2f) * imageW;
            float y1 = (cy - h / 2f) * imageH;
            float x2 = (cx + w / 2f) * imageW;
            float y2 = (cy + h / 2f) * imageH;

            float left   = clamp(Math.min(x1, x2), 0, imageW);
            float right  = clamp(Math.max(x1, x2), 0, imageW);
            float top    = clamp(Math.min(y1, y2), 0, imageH);
            float bottom = clamp(Math.max(y1, y2), 0, imageH);

            if (right - left < 2 || bottom - top < 2) continue;
            candidates.add(new BoundingBox(
                    new RectF(left, top, right, bottom),
                    bestScore,
                    bestClass
            ));
        }
//        return candidates;
        return nmsClassAware(candidates, IOU_THRESHOLD, MAX_DETECTIONS);
    }

    private static float get(float[] data, long[] shape, boolean channelsFirst, int row, int col) {
        int D1, D2;

        if(shape[1]<shape[2]){
            D1 = (int) shape[1];
            D2 = (int) shape[2];
        }else{
            D1 = (int) shape[2];
            D2 = (int) shape[1];
        }
        if (channelsFirst) {
            int index = (row * D2 + col) >= data.length ? data.length-1 : (row * D2 + col);
            // [1][row][col]
            return data[index];
        } else {
            int index = (col * D1 + row) >= data.length ? data.length-1 : (col * D1 + row);
            // [1][col][row]
            return data[index];
        }
    }

    public static List<BoundingBox> processFlatVulkan(
            float[] data,
            long[] shape,
            int inputSize,
            int imageW,
            int imageH
    ) {

        if (shape.length != 3) return new ArrayList<>();

        int numDetections = (int) shape[1]; // 300
        int valuesPerDet = (int) shape[2];  // 6

        if (valuesPerDet != 6) return new ArrayList<>();

        List<BoundingBox> boxes = new ArrayList<>();

        for (int i = 0; i < numDetections; i++) {

            int base = i * 6;

            float x1 = data[base];
            float y1 = data[base + 1];
            float x2 = data[base + 2];
            float y2 = data[base + 3];
            float score = data[base + 4];
            int cls = (int) data[base + 5];

            if (score < CONFIDENCE_THRESHOLD) continue;

            // ⚠️ Координаты уже в пикселях 640x640
            // Переводим к размеру текущего crop

            float scaleX = (float) imageW / inputSize;
            float scaleY = (float) imageH / inputSize;

            x1 *= scaleX;
            y1 *= scaleY;
            x2 *= scaleX;
            y2 *= scaleY;

            x1 = clamp(x1, 0, imageW);
            y1 = clamp(y1, 0, imageH);
            x2 = clamp(x2, 0, imageW);
            y2 = clamp(y2, 0, imageH);

            if (x2 - x1 < 2 || y2 - y1 < 2) continue;

            boxes.add(new BoundingBox(
                    new RectF(x1, y1, x2, y2),
                    score,
                    cls
            ));
        }

        return boxes; // NMS уже сделан моделью
    }
}
