package ru.nstu.navigator_arcore.tools;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Matrix;
import android.graphics.Rect;
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
    private static Bitmap workBitmap;
    private static Canvas workCanvas;
    private static Bitmap smallBitmap;
    private static Canvas smallCanvas;

    private static int[] argbSmall;
    private static byte[] ySmall;
    private static byte[] ySmallOut;

    private static StreamDenoiseY denoiser;

    private static final Rect srcRect = new Rect();
    private static final RectF dstRectF = new RectF();

    private static int clamp255(int v) {
        return v < 0 ? 0 : (v > 255 ? 255 : v);
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
    public static Bitmap noiseDetect(Bitmap[] images) {
        Bitmap original = images[images.length - 1];
        int w = original.getWidth();
        int h = original.getHeight();
        int cropSize = Math.min(w, h);
        int cropX = (w - cropSize) / 2;
        int cropY = (h - cropSize) / 2;

        final int DS = 2;
        int sw = Math.max(64, cropSize / DS);
        int sh = sw;

        if (workBitmap == null || workBitmap.getWidth() != w || workBitmap.getHeight() != h) {
            workBitmap = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888);
            workCanvas = new Canvas(workBitmap);
        }
        if (smallBitmap == null || smallBitmap.getWidth() != sw || smallBitmap.getHeight() != sh) {
            smallBitmap = Bitmap.createBitmap(sw, sh, Bitmap.Config.ARGB_8888);
            smallCanvas = new Canvas(smallBitmap);
        }
        if (argbSmall == null || argbSmall.length != sw * sh) {
            argbSmall = new int[sw * sh];
            ySmall = new byte[sw * sh];
            ySmallOut = new byte[sw * sh];
        }
        if (denoiser == null) {
            denoiser = new StreamDenoiseY(new StreamDenoiseY.Config());
        }
        workCanvas.drawBitmap(original, 0, 0, null);

        srcRect.set(cropX, cropY, cropX + cropSize, cropY + cropSize);
        dstRectF.set(0, 0, sw, sh);
        smallCanvas.drawBitmap(original, srcRect, dstRectF, null);
        smallBitmap.getPixels(argbSmall, 0, sw, 0, 0, sw, sh);
        for (int i = 0; i < argbSmall.length; i++) {
            int c = argbSmall[i];
            int r = (c >> 16) & 0xFF;
            int g = (c >> 8) & 0xFF;
            int b = c & 0xFF;
            int y = (77 * r + 150 * g + 29 * b) >> 8;   // fast luma
            ySmall[i] = (byte) y;
        }

        byte[] out = denoiser.process(ySmall, sw, sh);
        android.util.Log.d("DENOISE_MODE", "Mode: " + denoiser.getLastMode());
        if (out != ySmallOut) {
            System.arraycopy(out, 0, ySmallOut, 0, ySmallOut.length);
        }

        for (int i = 0; i < argbSmall.length; i++) {
            int c = argbSmall[i];
            int a = (c >>> 24);
            int r = (c >> 16) & 0xFF;
            int g = (c >> 8) & 0xFF;
            int b = c & 0xFF;

            int yin = ySmall[i] & 0xFF;
            int yout = ySmallOut[i] & 0xFF;
            int d = yout - yin;

            r = clamp255(r + d);
            g = clamp255(g + d);
            b = clamp255(b + d);

            argbSmall[i] = (a << 24) | (r << 16) | (g << 8) | b;
        }
        smallBitmap.setPixels(argbSmall, 0, sw, 0, 0, sw, sh);
        dstRectF.set(cropX, cropY, cropX + cropSize, cropY + cropSize);
        workCanvas.drawBitmap(smallBitmap, null, dstRectF, null);

        return workBitmap;
    }
}
