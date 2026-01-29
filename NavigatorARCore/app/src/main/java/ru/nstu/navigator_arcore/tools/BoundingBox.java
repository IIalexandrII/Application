package ru.nstu.navigator_arcore.tools;

import android.graphics.RectF;

public class BoundingBox {

    // Прямоугольник объекта на экране (координаты в пикселях/нормализованные — как придут из детектора)
    public RectF rect;

    // Уверенность нейросети в этом объекте (0..1)
    public float score;

    // Класс объекта (id категории из модели)
    public int clazz;

    // Расстояние до объекта из ARCore depth (в метрах).
    // Может быть null, если глубина недоступна.
    public Float distanceMeters;

    public BoundingBox(RectF rect, float score, int clazz) {
        // Сохраняю геометрию рамки
        this.rect = rect;

        // Сохраняю confidence детекции
        this.score = score;

        // Сохраняю класс объекта
        this.clazz = clazz;
    }
}
