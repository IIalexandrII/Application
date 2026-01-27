package ru.nstu.navigator_arcore;

import ru.nstu.navigator_arcore.tools.BoundingBox;
import ru.nstu.navigator_arcore.tools.PostProcessor;

import android.app.Activity;
import android.content.Context;
import android.media.Image;
import android.util.Log;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;


public class Model {
    Module module;
    Activity activity;
    public String[] Classes;
//    Model(String fileNameModel, String fileNameClasses, Context context) throws Exception {
//        if (!(context instanceof Activity)) {
//            Log.e("Error LOG.E", "class Model (constructor): context is not Activity");
//            throw new Exception("class Model (constructor): context is not Activity");
//        }
//        this.activity = (Activity) context;
//        try {
//            this._loadModelFromFile(fileNameModel);
//            this._loadClassesFromFile(fileNameClasses);
//        } catch (Exception e) {
//            Log.e("Error LOG.E", "Model (Constructor): " + e.getMessage());
//        }
//    }

    Model(String modelPath, String classesPath, Context context) throws Exception {
        if (!(context instanceof Activity)) {
            Log.e("Error LOG.E", "class Model (constructor): context is not Activity");
            throw new Exception("Context is not Activity");
        }
        this.activity = (Activity) context;

        this._loadModelFromPath(modelPath);
        this._loadClassesFromPath(classesPath);
    }

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

    private void _loadModelFromFile(String fileName) throws IOException {
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
    private void _loadClassesFromFile(String fileName) throws IOException {
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

    public List<BoundingBox> analyzeImage(Image image, int rotation) {
        Tensor input = TensorImageUtils.imageYUV420CenterCropToFloat32Tensor(
                image,
                rotation,
                640, 640,
                TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,
                TensorImageUtils.TORCHVISION_NORM_STD_RGB
        );

        Tensor out = module.forward(IValue.from(input)).toTensor();

        float[] data = out.getDataAsFloatArray();
        long[] shape = out.shape();

        return PostProcessor.process(
                data,
                shape,
                640,
                image.getWidth(),
                image.getHeight()
        );
    }

}
