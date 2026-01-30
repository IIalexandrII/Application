package ru.nstu.navigator_arcore.modelTools;


import android.graphics.Bitmap;

import org.pytorch.executorch.Tensor;

import java.nio.FloatBuffer;

public class TensorImageUtils {
    private TensorImageUtils(){}

    public static final float[] TORCHVISION_NORM_MEAN_RGB = new float[]{0.485f, 0.456f, 0.406f};
    public static final float[] TORCHVISION_NORM_STD_RGB = new float[]{0.229f, 0.224f, 0.225f};

    private static void checkNormStdArg(float[] normStdRGB) {
        if (normStdRGB == null || normStdRGB.length != 3) {
            throw new IllegalArgumentException("normStdRGB length must be 3");
        }
    }
    private static void checkNormMeanArg(float[] normMeanRGB) {
        if (normMeanRGB == null || normMeanRGB.length != 3) {
            throw new IllegalArgumentException("normMeanRGB length must be 3");
        }
    }

    public static Tensor bitmapToFloat32Tensor(Bitmap bitmap, float[] normMeanRGB, float[] normStdRGB) {
        checkNormMeanArg(normMeanRGB);
        checkNormStdArg(normStdRGB);
        return _bitmapToFloat32Tensor(
                bitmap,
                0, 0,
                bitmap.getWidth(), bitmap.getHeight(),
                normMeanRGB, normStdRGB
        );
    }

    private static Tensor _bitmapToFloat32Tensor(
            Bitmap bitmap,
            int x, int y,
            int width, int height,
            float[] normMeanRGB, float[] normStdRGB
    ) {
        FloatBuffer floatBuffer = Tensor.allocateFloatBuffer(3 * width * height);

        _bitmapToFloatBuffer(
                bitmap,
                x, y,
                width, height,
                normMeanRGB, normStdRGB,
                floatBuffer,
                0
        );

        return Tensor.fromBlob(floatBuffer, new long[]{1, 3, height, width});
    }

    private static void _bitmapToFloatBuffer(
            Bitmap bitmap,
            int x, int y,
            int width, int height,
            float[] normMeanRGB, float[] normStdRGB,
            FloatBuffer outBuffer, int outBufferOffset
    ) {
        int pixelsCount = width * height;
        int[] pixels = new int[pixelsCount];

        bitmap.getPixels(pixels, 0, width, x, y, width, height);

        int offsetG = pixelsCount;
        int offsetB = 2 * pixelsCount;

        for (int i = 0; i < pixelsCount; i++) {
            int c = pixels[i];

            float r = ((c >> 16) & 0xff) / 255.0f;
            float g = ((c >> 8) & 0xff) / 255.0f;
            float b = (c & 0xff) / 255.0f;

            outBuffer.put(outBufferOffset + i, (r - normMeanRGB[0]) / normStdRGB[0]);
            outBuffer.put(outBufferOffset + offsetG + i, (g - normMeanRGB[1]) / normStdRGB[1]);
            outBuffer.put(outBufferOffset + offsetB + i, (b - normMeanRGB[2]) / normStdRGB[2]);
        }
    }
}
