package ru.nstu.navigator_arcore.conductorTools;

import android.content.Context;
import android.media.Image;
import android.os.VibrationEffect;
import android.os.Vibrator;

import java.nio.ByteBuffer;

public class DepthObstacleDetector {

    private Vibrator vibrator;

    private long lastVibration = 0;

    private static final float MAX_DISTANCE = 2.5f;
    private static final float MIN_DISTANCE = 0.4f;

    public DepthObstacleDetector(Context context){
        vibrator = (Vibrator) context.getSystemService(Context.VIBRATOR_SERVICE);
    }

    public void processDepth(Image depthImage){

        if(depthImage == null) return;

        int w = depthImage.getWidth();
        int h = depthImage.getHeight();

        Image.Plane plane = depthImage.getPlanes()[0];
        ByteBuffer buffer = plane.getBuffer();

        int rowStride = plane.getRowStride();
        int pixelStride = plane.getPixelStride();

        float nearest = Float.MAX_VALUE;

        // анализ центральной области (как в DepthScanner)
        int startX = w/3;
        int endX = w*2/3;

        int startY = h/3;
        int endY = h*2/3;

        for(int y=startY;y<endY;y+=4){
            for(int x=startX;x<endX;x+=4){

                int index = y * rowStride + x * pixelStride;

                int lo = buffer.get(index) & 0xFF;
                int hi = buffer.get(index+1) & 0xFF;

                int mm = (hi << 8) | lo;

                if(mm <= 0) continue;

                float m = mm / 1000f;

                if(m < nearest)
                    nearest = m;
            }
        }

        if(nearest < MAX_DISTANCE){
            vibrate(nearest);
        }
    }

    private void vibrate(float distance){

        long now = System.currentTimeMillis();

        // частота зависит от расстояния
        long interval;

        if(distance < 0.5)
            interval = 100;
        else if(distance < 1)
            interval = 250;
        else if(distance < 2)
            interval = 600;
        else
            interval = 1200;

        if(now - lastVibration > interval){

            vibrator.vibrate(
                    VibrationEffect.createOneShot(
                            80,
                            VibrationEffect.DEFAULT_AMPLITUDE
                    )
            );

            lastVibration = now;
        }
    }
}
