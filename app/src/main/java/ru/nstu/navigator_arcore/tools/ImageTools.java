package ru.nstu.navigator_arcore.tools;

import android.graphics.Bitmap;
import android.graphics.Matrix;
import android.graphics.RectF;
import android.util.Log;
import android.view.Surface;

public class ImageTools {
    private ImageTools(){}

    public static int getTotalRotation(int surfaceRotation) {
        switch (surfaceRotation) {
            case Surface.ROTATION_0:   return 90;
            case Surface.ROTATION_90:  return 0;
            case Surface.ROTATION_180: return 270;
            case Surface.ROTATION_270: return 180;
        }
        return 90;
    }

    public static Bitmap rotateBitmap(Bitmap src, int rotation) {
        int totalRotation = getTotalRotation(rotation);

        Matrix m = new Matrix();
        m.postRotate(totalRotation);
        return Bitmap.createBitmap(src, 0, 0, src.getWidth(), src.getHeight(), m, true);
    }

    public static RectF undoRotate(RectF r, int srcW, int srcH, int totalRotation) {
        switch (totalRotation) {
            case 0:   return new RectF(r);
            case 90:  return new RectF(r.top, srcH - r.right, r.bottom, srcH - r.left);
            case 180: return new RectF(srcW - r.right, srcH - r.bottom, srcW - r.left, srcH - r.top);
            case 270: return new RectF(srcW - r.bottom, r.left, srcW - r.top, r.right);
        }
        return new RectF(r);
    }
    // Размер входящего буфера сейчас 5.
    // Определяется полем bufferSize в классе ARCoreRenderer.
    // Все bitmap'ы уже переведены в RGB и повернуты правильно
    // Последее изображение массива является текущим которое попадет на YOLO (Images.length - 1)
    public static Bitmap noiseDetect(Bitmap[] Images){
        return Images[Images.length - 1];
    }
}
