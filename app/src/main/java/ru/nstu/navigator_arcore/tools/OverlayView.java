package ru.nstu.navigator_arcore.tools;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;
import android.util.AttributeSet;
import android.view.View;

import androidx.annotation.NonNull;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

public class OverlayView extends View {

    // Кисть для рамок вокруг объектов
    private final Paint boxPaint = new Paint();

    // Кисть для подписей (класс, confidence, дистанция)
    private final Paint textPaint = new Paint();

    // Текущий список найденных объектов
    private List<BoundingBox> boxes = new ArrayList<>();

    // Названия классов из модели
    private String[] classes = new String[0];

    public OverlayView(Context context, AttributeSet attrs) {
        super(context, attrs);

        // Настройка рамки: зеленая обводка, без заливки
        boxPaint.setStyle(Paint.Style.STROKE);
        boxPaint.setStrokeWidth(6f);
        boxPaint.setColor(Color.GREEN);

        // Настройка текста: тем же цветом, чтобы выглядело единообразно
        textPaint.setColor(Color.GREEN);
        textPaint.setTextSize(42f);
        textPaint.setStyle(Paint.Style.FILL);
    }

    // Обновляю результаты детекции и прошу View перерисоваться
    public void setResults(List<BoundingBox> boxes, String[] classes) {
        this.boxes = (boxes != null) ? boxes : new ArrayList<>();
        this.classes = (classes != null) ? classes : new String[0];
        invalidate(); // триггер onDraw()
    }

    @Override
    protected void onDraw(@NonNull Canvas canvas) {
        super.onDraw(canvas);

        // Прохожусь по всем найденным объектам и рисую поверх камеры
        for (BoundingBox box : boxes) {
            if (box == null || box.rect == null) continue;

            // Копирую прямоугольник, чтобы не портить исходные данные
            RectF r = new RectF(box.rect);

            // Нормализую координаты на всякий случай
            // (иногда детектор может отдать перевернутые стороны)
            if (r.left > r.right) {
                float t = r.left;
                r.left = r.right;
                r.right = t;
            }
            if (r.top > r.bottom) {
                float t = r.top;
                r.top = r.bottom;
                r.bottom = t;
            }

            // Слишком маленькие боксы не рисую — это просто шум
            if (r.width() < 2 || r.height() < 2) continue;

            // Рисую рамку объекта
            canvas.drawRect(r, boxPaint);

            // Получаю имя класса по id
            String name = (box.clazz >= 0 && box.clazz < classes.length)
                    ? classes[box.clazz]
                    : ("cls " + box.clazz);

            // Формирую подпись: класс + confidence
            String label = name + String.format(Locale.US, " %.2f", box.score);

            // Если есть глубина — добавляю расстояние в метрах
            if (box.distanceMeters != null) {
                label += String.format(Locale.US, " %.2fm", box.distanceMeters);
            }

            // Рисую текст чуть выше рамки, чтобы не перекрывать объект
            canvas.drawText(label, r.left, Math.max(60f, r.top - 12f), textPaint);
        }
    }
}
