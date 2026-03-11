package ru.nstu.navigator_arcore.tools;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.ImageFormat;
import android.media.Image;

import android.renderscript.Allocation;
import android.renderscript.Element;
import android.renderscript.RenderScript;
import android.renderscript.ScriptIntrinsicYuvToRGB;
import android.renderscript.Type;

public class YuvToRgbConverter {
    // RenderScript использую как быстрый конвертер из YUV в RGB.
    // Да, он deprecated, но для проекта работает нормально.
    private final RenderScript rs;
    private final ScriptIntrinsicYuvToRGB yuvToRgbIntrinsic;

    // Allocation'ы держу между кадрами, чтобы не создавать их постоянно
    private Allocation in;
    private Allocation out;

    // Буфер NV21 (Y + VU), тоже переиспользую
    private byte[] nv21;

    public YuvToRgbConverter(Context context) {
        rs = RenderScript.create(context.getApplicationContext());
        yuvToRgbIntrinsic = ScriptIntrinsicYuvToRGB.create(rs, Element.U8_4(rs));
    }

    // Конверчу Image (YUV_420_888) в Bitmap (ARGB_8888)
    public void yuvToRgb(Image image, Bitmap output) {
        // Тут ожидаю только YUV_420_888
        if (image.getFormat() != ImageFormat.YUV_420_888) {
            throw new IllegalArgumentException("Expected YUV_420_888, got: " + image.getFormat());
        }

        int w = image.getWidth();
        int h = image.getHeight();

        // NV21 по размеру: Y (w*h) + VU (w*h/2)
        int ySize = w * h;
        int uvSize = w * h / 2;
        int total = ySize + uvSize;

        // Подстраиваю буфер под размер кадра (обычно он не меняется)
        if (nv21 == null || nv21.length != total) {
            nv21 = new byte[total];
        }

        // Камера дает 3 plane: Y, U, V. Собираю их в NV21.
        Image.Plane yPlane = image.getPlanes()[0];
        Image.Plane uPlane = image.getPlanes()[1];
        Image.Plane vPlane = image.getPlanes()[2];

        // Y копирую как есть
        copyPlane(yPlane, w, h, nv21, 0);

        // UV собираю как VU (NV21 именно так хочет)
        interleaveVU(uPlane, vPlane, w, h, nv21, ySize);

        // Allocation'ы создаю один раз под нужный размер
        if (in == null) {
            Type.Builder yuvType = new Type.Builder(rs, Element.U8(rs)).setX(total);
            in = Allocation.createTyped(rs, yuvType.create(), Allocation.USAGE_SCRIPT);

            Type.Builder rgbaType = new Type.Builder(rs, Element.RGBA_8888(rs)).setX(w).setY(h);
            out = Allocation.createTyped(rs, rgbaType.create(), Allocation.USAGE_SCRIPT);
        }

        // Конвертация через intrinsic
        in.copyFrom(nv21);
        yuvToRgbIntrinsic.setInput(in);
        yuvToRgbIntrinsic.forEach(out);

        // Готовый результат копирую в Bitmap
        out.copyTo(output);
    }

    // Копирую один plane в плоский массив
    private static void copyPlane(Image.Plane plane, int width, int height, byte[] out, int offset) {
        java.nio.ByteBuffer buf = plane.getBuffer();
        int rowStride = plane.getRowStride();
        int pixelStride = plane.getPixelStride();

        byte[] row = new byte[rowStride];

        int outPos = offset;
        for (int y = 0; y < height; y++) {
            int rowStart = y * rowStride;
            buf.position(rowStart);
            buf.get(row, 0, Math.min(rowStride, buf.remaining()));

            if (pixelStride == 1) {
                // обычно так и бывает
                System.arraycopy(row, 0, out, outPos, width);
                outPos += width;
            } else {
                // редко, но лучше обработать
                for (int x = 0; x < width; x++) {
                    out[outPos++] = row[x * pixelStride];
                }
            }
        }
    }

    // Собираю UV часть в формате NV21: VU VU VU...
    private static void interleaveVU(Image.Plane uPlane, Image.Plane vPlane, int width, int height, byte[] out, int offset) {
        java.nio.ByteBuffer uBuf = uPlane.getBuffer();
        java.nio.ByteBuffer vBuf = vPlane.getBuffer();

        int uvW = width / 2;
        int uvH = height / 2;

        int uRowStride = uPlane.getRowStride();
        int vRowStride = vPlane.getRowStride();
        int uPixelStride = uPlane.getPixelStride();
        int vPixelStride = vPlane.getPixelStride();

        byte[] uRow = new byte[uRowStride];
        byte[] vRow = new byte[vRowStride];

        int outPos = offset;

        for (int y = 0; y < uvH; y++) {
            int uRowStart = y * uRowStride;
            int vRowStart = y * vRowStride;

            uBuf.position(uRowStart);
            vBuf.position(vRowStart);

            uBuf.get(uRow, 0, Math.min(uRowStride, uBuf.remaining()));
            vBuf.get(vRow, 0, Math.min(vRowStride, vBuf.remaining()));

            for (int x = 0; x < uvW; x++) {
                byte U = uRow[x * uPixelStride];
                byte V = vRow[x * vPixelStride];

                // порядок важен: сначала V, потом U
                out[outPos++] = V;
                out[outPos++] = U;
            }
        }
    }
}
