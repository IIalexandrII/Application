package ru.nstu.navigator_arcore.tools;

import android.graphics.Bitmap;
import android.graphics.Matrix;
import android.util.Log;
import android.view.Surface;

public class ImageTools {
    private ImageTools(){}
    public static Bitmap rotateBitmap(Bitmap src, int rotation) {
        Matrix m = new Matrix();
        m.postRotate(90);

        if (rotation == Surface.ROTATION_0) return Bitmap.createBitmap(src, 0, 0, src.getWidth(), src.getHeight(), m, true);
        Log.d("TESTTEST", ""+rotation);
        switch (rotation) {
            case Surface.ROTATION_90:
                m.postRotate(270);
                break;
            case Surface.ROTATION_180:
                m.postRotate(180);
                break;
            case Surface.ROTATION_270:
                m.postRotate(90);
                break;
        }

        return Bitmap.createBitmap(src, 0, 0, src.getWidth(), src.getHeight(), m, true);
    }

    public static Bitmap noiseDetect(Bitmap Image){
        return Image;
    }
}
