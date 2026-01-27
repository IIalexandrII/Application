package ru.nstu.navigator_arcore.tools;

import android.graphics.RectF;

import androidx.annotation.Nullable;

public class BoundingBox {
    public RectF rect;
    public float score;
    public int clazz;

    public Float distanceMeters;

    public BoundingBox(RectF rect, float score, int clazz, @Nullable Float distanceMeters) {
        this.rect = rect;
        this.score = score;
        this.clazz = clazz;
        this.distanceMeters = distanceMeters;
    }
}
