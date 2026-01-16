package ru.nstu.navigator.tools;

import android.graphics.RectF;
public class BoundingBox {
    public RectF rect;
    public float score;
    public int clazz;

    public BoundingBox(RectF rect, float score, int clazz) {
        this.rect = rect;
        this.score = score;
        this.clazz = clazz;
    }
}
