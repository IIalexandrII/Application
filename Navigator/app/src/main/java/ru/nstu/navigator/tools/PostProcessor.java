package ru.nstu.navigator.tools;

import org.pytorch.Tensor;
import android.graphics.RectF;
import android.util.Log;

import java.util.ArrayList;
import java.util.List;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;

public class PostProcessor {
    private static final float CONFIDENCE_THRESHOLD = 0.25f;
    private static final float IOU_THRESHOLD = 0.4f;
    private static final int MAX_NMS = 3000;

    public static List<BoundingBox> process(Tensor tensor, float scaleX, float scaleY) {
        return nms(tensor, CONFIDENCE_THRESHOLD, IOU_THRESHOLD, MAX_NMS, scaleX, scaleY);
    }

    public static List<BoundingBox> nms(Tensor tensor,
                                        float confidenceThreshold,
                                        float iouThreshold,
                                        int maxNms,
                                        float scaleX,
                                        float scaleY) {
        float[] array = tensor.getDataAsFloatArray();
        long[] shape = tensor.shape();
        Log.d("YOLO", String.valueOf(array.length));
        Log.d("YOLO", String.valueOf(shape.length));

        Log.d("YOLO", array[0] + " " + array[1] + " " + array[2] + " " + array[3] + " " + array[4]);
        Log.d("YOLO", shape[0] + " " + shape[1] + " " + shape[2]);


        int rows = (int) shape[1];
        int cols = (int) shape[2];

        List<BoundingBox> results = new ArrayList<>();

        for (int col = 0; col < cols; col++) {
            float maxScore = 0;
            int maxClass = 0;

            for (int classRow = 4; classRow < rows; classRow++) {
                float score = array[classRow * cols + col];
                if (score > maxScore) {
                    maxScore = score;
                    maxClass = classRow - 4;
                }
            }

            if (maxScore > confidenceThreshold) {
                float x = array[0 * cols + col];
                float y = array[1 * cols + col];
                float w = array[2 * cols + col];
                float h = array[3 * cols + col];

                float left = x - w / 2;
                float top = y - h / 2;
                float right = x + w / 2;
                float bottom = y + h / 2;

                RectF rect = new RectF(
                        left * scaleX,
                        top * scaleY,
                        right * scaleX,
                        bottom * scaleY
                );

                results.add(new BoundingBox(rect, maxScore, maxClass));
            }
        }

        return nms(results, iouThreshold, maxNms);
    }

    private static List<BoundingBox> nms(List<BoundingBox> boxes, float iouThreshold, int limit) {
        if (boxes.isEmpty()) {
            return new ArrayList<>();
        }

        List<BoundingBox> sortedBoxes = new ArrayList<>(boxes);
        Collections.sort(sortedBoxes, new Comparator<BoundingBox>() {
            @Override
            public int compare(BoundingBox o1, BoundingBox o2) {
                return Float.compare(o2.score, o1.score);
            }
        });

        boolean[] active = new boolean[sortedBoxes.size()];
        Arrays.fill(active, true);
        int numActive = active.length;

        List<BoundingBox> selected = new ArrayList<>();

        for (int i = 0; i < sortedBoxes.size(); i++) {
            if (!active[i]) continue;

            BoundingBox boxA = sortedBoxes.get(i);
            selected.add(boxA);

            if (selected.size() >= limit) break;

            for (int j = i + 1; j < sortedBoxes.size(); j++) {
                if (active[j]) {
                    BoundingBox boxB = sortedBoxes.get(j);

                    if (boxA.clazz == boxB.clazz) {
                        float iou = calculateIoU(boxA.rect, boxB.rect);
                        if (iou > iouThreshold) {
                            active[j] = false;
                            numActive--;
                            if (numActive <= 0) {
                                return selected;
                            }
                        }
                    }
                }
            }
        }

        return selected;
    }

    private static float calculateIoU(RectF a, RectF b) {
        float areaA = (a.right - a.left) * (a.bottom - a.top);
        if (areaA <= 0) return 0.0f;

        float areaB = (b.right - b.left) * (b.bottom - b.top);
        if (areaB <= 0) return 0.0f;

        float interLeft = Math.max(a.left, b.left);
        float interTop = Math.max(a.top, b.top);
        float interRight = Math.min(a.right, b.right);
        float interBottom = Math.min(a.bottom, b.bottom);

        float interWidth = Math.max(interRight - interLeft, 0);
        float interHeight = Math.max(interBottom - interTop, 0);
        float interArea = interWidth * interHeight;

        float unionArea = areaA + areaB - interArea;

        if (unionArea <= 0) return 0.0f;

        return interArea / unionArea;
    }
}