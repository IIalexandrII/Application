package ru.nstu.navigator.tools;

import android.content.Context;
import android.graphics.Color;
import android.graphics.Paint;
import android.util.AttributeSet;
import android.view.View;
import android.graphics.Canvas;

import androidx.annotation.NonNull;

import java.util.ArrayList;
import java.util.List;

public class OverlayView extends View {
    private final Paint boxPaint = new Paint();
    private final Paint textPaint = new Paint();
    private List<BoundingBox> boxes = new ArrayList<>();
    private String[] classes = new String[0];

    public OverlayView(Context context, AttributeSet attrs) {
        super(context, attrs);

        boxPaint.setStyle(Paint.Style.STROKE);
        boxPaint.setStrokeWidth(5f);
        boxPaint.setColor(Color.GREEN);

        textPaint.setColor(Color.GREEN);
        textPaint.setTextSize(40f);
        textPaint.setStyle(Paint.Style.FILL);
    }

    public void setResults(List<BoundingBox> boxes, String[] classes) {
        this.boxes = boxes;
        this.classes = classes;
        invalidate();
    }

    @Override
    protected void onDraw(@NonNull Canvas canvas) {
        super.onDraw(canvas);

        for (BoundingBox box : boxes) {
            canvas.drawRect(box.rect, boxPaint);

            String label = classes[box.clazz] + String.format(" %.2f", box.score);

            canvas.drawText(
                    label,
                    box.rect.left,
                    box.rect.top - 10,
                    textPaint
            );
        }
    }
}
