package ru.nstu.navigator_arcore.tools;

import android.graphics.RectF;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

public class PostProcessor {
    private static final float CONFIDENCE_THRESHOLD = 0.10f;
    private static final float IOU_THRESHOLD = 0.45f;
    private static final int MAX_DETECTIONS = 50;

    public static List<BoundingBox> process(float[] data, long[] shape, int inputSize, int imageW, int imageH) {
        if (data == null || shape == null || shape.length != 3) return new ArrayList<>();

        final int dim1 = (int) shape[1];
        final int dim2 = (int) shape[2];

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
            channelsFirst = true;
            rows = dim1;
            cols = dim2;
        }

        if (rows < 5) return new ArrayList<>();
        final int numClasses = rows - 4;

        float maxCoord = 0f;
        int probe = Math.min(cols, 200);
        for (int i = 0; i < probe; i++) {
            float cx = get(data, channelsFirst, rows, cols, 0, i);
            float cy = get(data, channelsFirst, rows, cols, 1, i);
            float w  = get(data, channelsFirst, rows, cols, 2, i);
            float h  = get(data, channelsFirst, rows, cols, 3, i);
            maxCoord = Math.max(maxCoord, Math.max(Math.max(cx, cy), Math.max(w, h)));
        }

        final boolean coordsArePixels = maxCoord > 2.5f;

        List<BoundingBox> candidates = new ArrayList<>();

        for (int i = 0; i < cols; i++) {
            float cx = get(data, channelsFirst, rows, cols, 0, i);
            float cy = get(data, channelsFirst, rows, cols, 1, i);
            float w  = get(data, channelsFirst, rows, cols, 2, i);
            float h  = get(data, channelsFirst, rows, cols, 3, i);

            int bestClass = -1;
            float bestScore = 0f;

            for (int c = 0; c < numClasses; c++) {
                float s = get(data, channelsFirst, rows, cols, 4 + c, i);
                if (s > bestScore) {
                    bestScore = s;
                    bestClass = c;
                }
            }
            if (bestScore < CONFIDENCE_THRESHOLD) continue;

            if (coordsArePixels) {
                cx /= (float) inputSize;
                cy /= (float) inputSize;
                w  /= (float) inputSize;
                h  /= (float) inputSize;
            }

            float x1 = (cx - w / 2f) * imageW;
            float y1 = (cy - h / 2f) * imageH;
            float x2 = (cx + w / 2f) * imageW;
            float y2 = (cy + h / 2f) * imageH;

            float left   = Math.min(x1, x2);
            float right  = Math.max(x1, x2);
            float top    = Math.min(y1, y2);
            float bottom = Math.max(y1, y2);

            left =   clamp(left, 0f, imageW);
            right =  clamp(right, 0f, imageW);
            top =    clamp(top, 0f, imageH);
            bottom = clamp(bottom, 0f, imageH);

            if ((right - left) < 2f || (bottom - top) < 2f) continue;

            candidates.add(new BoundingBox(new RectF(left, top, right, bottom), bestScore, bestClass, null));
        }

        return nmsClassAware(candidates, IOU_THRESHOLD, MAX_DETECTIONS);
    }

    private static float get(
            float[] data,
            boolean channelsFirst,
            int rows,
            int cols,
            int row,
            int col
    ) {
        // shape = [1][rows][cols] или [1][cols][rows]
        int index;
        if (channelsFirst) {
            index = row * cols + col;
        } else {
            index = col * rows + row;
        }
        return data[index];
    }
    private static List<BoundingBox> nmsClassAware(List<BoundingBox> boxes, float iouThreshold, int limit) {
        if (boxes.isEmpty()) return new ArrayList<>();

        Collections.sort(boxes, new Comparator<BoundingBox>() {
            @Override public int compare(BoundingBox a, BoundingBox b) {
                return Float.compare(b.score, a.score);
            }
        });

        List<BoundingBox> selected = new ArrayList<>();
        for (BoundingBox a : boxes) {
            boolean keep = true;

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

        return inter / (union + 1e-6f);
    }

    private static float clamp(float v, float lo, float hi) {
        return Math.max(lo, Math.min(hi, v));
    }
}
