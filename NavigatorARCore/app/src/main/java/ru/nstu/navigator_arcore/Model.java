package ru.nstu.navigator_arcore;

import android.app.Activity;
import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Rect;
import android.media.Image;
import android.util.Log;

import org.pytorch.executorch.EValue;
import org.pytorch.executorch.Module;
import org.pytorch.executorch.Tensor;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import ru.nstu.navigator_arcore.modelTools.TensorImageUtils;
import ru.nstu.navigator_arcore.modelTools.YuvToRgbConverter;
import ru.nstu.navigator_arcore.tools.BoundingBox;
import ru.nstu.navigator_arcore.tools.PostProcessor;

public class Model {
    Module module;
    public String[] Classes;

    private static final int IN_W = 640;
    private static final int IN_H = 640;

    private Bitmap rgbBitmap = null;
    private Bitmap inputBitmap = null;
    private Canvas inputCanvas = null;

    private final Rect srcRect = new Rect();
    private final Rect dstRect = new Rect(0, 0, IN_W, IN_H);

    private final YuvToRgbConverter yuvToRgb;

    private final Activity activity;
    Model(String modelPath, String classesPath, Context context, boolean useAssets) throws Exception {
        if (!(context instanceof Activity)) {
            Log.e("Error LOG.E", "class Model (constructor): context is not Activity");
            throw new Exception("Context is not Activity");
        }
        this.activity = (Activity) context;

        if (useAssets){
            try {
                this._loadModelFromAssets(modelPath);
                this._loadClassesFromAssets(classesPath);
            } catch (Exception e) {
                Log.e("Error LOG.E", "Model (Constructor): " + e.getMessage());
            }
        }else {
            this._loadModelFromPath(modelPath);
            this._loadClassesFromPath(classesPath);
        }

        this.yuvToRgb = new YuvToRgbConverter(context);

        inputBitmap = Bitmap.createBitmap(IN_W, IN_H, Bitmap.Config.ARGB_8888);
        inputCanvas = new Canvas(inputBitmap);
    }

    // LOAD MODEL -----------------------------------------------
    private void _loadModelFromPath(String path) throws IOException {
        File modelFile = new File(path);
        if (!modelFile.exists()) {
            throw new IOException("Model file not found: " + path);
        }
        module = Module.load(modelFile.getAbsolutePath());
    }
    private void _loadClassesFromPath(String path) throws IOException {
        try (BufferedReader reader = new BufferedReader(new FileReader(path))) {
            String line = reader.readLine();
            if (line == null) throw new IOException("Empty classes file");

            Classes = line.split(",");
            for (int i = 0; i < Classes.length; i++) {
                Classes[i] = Classes[i].trim();
            }
        }
    }

    private void _loadModelFromAssets(String fileName) throws IOException {
        File modelFile = new File(this.activity.getFilesDir(), fileName);

        if (!modelFile.exists()) {
            try (InputStream inputStream = this.activity.getAssets().open(fileName);
                 FileOutputStream outputStream = new FileOutputStream(modelFile)) {

                byte[] buffer = new byte[2048];
                int bytesRead;
                long totalBytes = 0;

                while ((bytesRead = inputStream.read(buffer)) != -1) {
                    outputStream.write(buffer, 0, bytesRead);
                    totalBytes += bytesRead;
                }

            } catch (IOException e) {
                Log.e("Error LOG.E", "Model (_loadModelFromFile): " + e.getMessage());
                throw e;
            }
        }

        try {
            module = Module.load(modelFile.getAbsolutePath());
        } catch (Exception e) {
            Log.e("Model", "Failed to load PyTorch model", e);
            throw new IOException("Failed to load PyTorch model: " + e.getMessage());
        }
    }
    private void _loadClassesFromAssets(String fileName) throws IOException {
        try (InputStream inputStream = this.activity.getAssets().open(fileName); BufferedReader reader = new BufferedReader(new InputStreamReader(inputStream))) {
            String line;

            while ((line = reader.readLine()) != null) {
                line = line.trim();
                if (!line.isEmpty()) {
                    String[] classArray = line.split(",");
                    List<String> classList = new ArrayList<>();
                    for (String className : classArray) {
                        className = className.trim();
                        if (!className.isEmpty()) {
                            classList.add(className);
                        }
                    }
                    Classes = classList.toArray(new String[0]);
                    break;
                }
            }

            if (Classes == null || Classes.length == 0) {
                Log.e("Error LOG.E", "Model (_loadClassesFromFile): Failed to load CLASSES: file is empty or contains no valid data" );
                throw new IOException("Failed to load CLASSES: file is empty or contains no valid data");
            }

        }
    }
// ----------------------------------------------------------

    public List<BoundingBox> analyzeImage(Image image, int rotation) {
        final int camW = image.getWidth();
        final int camH = image.getHeight();

        if (rgbBitmap == null || rgbBitmap.getWidth() != camW || rgbBitmap.getHeight() != camH) {
            rgbBitmap = Bitmap.createBitmap(camW, camH, Bitmap.Config.ARGB_8888);
        }

        yuvToRgb.yuvToRgb(image, rgbBitmap);

        int cropSize = Math.min(camW, camH);
        int cropX = (camW - cropSize) / 2;
        int cropY = (camH - cropSize) / 2;

        srcRect.set(cropX, cropY, cropX + cropSize, cropY + cropSize);

        inputCanvas.drawBitmap(rgbBitmap, srcRect, dstRect, null);

        Tensor input = TensorImageUtils.bitmapToFloat32Tensor(inputBitmap, TensorImageUtils.TORCHVISION_NORM_MEAN_RGB, TensorImageUtils.TORCHVISION_NORM_STD_RGB);

        Tensor out = module.forward(EValue.from(input))[0].toTensor();

        float[] data = out.getDataAsFloatArray();
        long[] shape = out.shape();

        List<BoundingBox> tempBoxes = PostProcessor.processFlat(data, shape, 640, cropSize, cropSize);

        for (BoundingBox b : tempBoxes) {
            if (b == null || b.rect == null) continue;

            b.rect.offset(cropX, cropY);

            if (b.rect.left < 0) b.rect.left = 0;
            if (b.rect.top < 0) b.rect.top = 0;
            if (b.rect.right > camW) b.rect.right = camW;
            if (b.rect.bottom > camH) b.rect.bottom = camH;
        }

        return tempBoxes;
    }
}
