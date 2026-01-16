package ru.nstu.navigator;

import android.app.Activity;
import android.content.Context;
import android.util.Log;

import androidx.annotation.OptIn;
import androidx.camera.core.ExperimentalGetImage;
import androidx.camera.core.ImageProxy;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.List;
import java.util.Objects;

import ru.nstu.navigator.tools.BoundingBox;
import ru.nstu.navigator.tools.PostProcessor;

public class Model {
    Module module;
    Activity activity;
    String[] Classes = new String[] {"Light", "Trafic", "Person", "Scooter", "Open_Manhole", "Door", "Stairs", "Car", "Bus", "Bicycle"};
    Model(String fileName, Context context) throws Exception {
        if (!(context instanceof Activity)){
            Log.e("TORCH", "context is not Activity");
            throw new Exception("class Model (constructor): context is not Activity");
        }
        this.activity = (Activity) context;
        File modelFile = new File(this.activity.getFilesDir(), fileName);

        try{
            if(!modelFile.exists()){
                InputStream inputStream = this.activity.getAssets().open(fileName);
                FileOutputStream outputStream = new FileOutputStream(modelFile);
                byte[] buffer = new byte[2048];
                int bytesReed = -1;
                while((bytesReed = inputStream.read(buffer)) != -1){
                    outputStream.write(buffer,0,bytesReed);
                }
                inputStream.close();
                outputStream.close();
            }
            Log.d("TORCH", "PATH: "+modelFile.getAbsolutePath());
            module = Module.load(modelFile.getAbsolutePath());

        }catch (IOException e){
            e.printStackTrace();
        }

    }

    @OptIn(markerClass = ExperimentalGetImage.class)
    List<BoundingBox> analyzeImage(ImageProxy image, int rotation){
        Tensor inputTensor = TensorImageUtils.imageYUV420CenterCropToFloat32Tensor(
                Objects.requireNonNull(image.getImage()),
                rotation,
                640,640,
                TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,
                TensorImageUtils.TORCHVISION_NORM_STD_RGB
        );

        Tensor outputTensor = module.forward(IValue.from(inputTensor)).toTensor();
        float scaleX = image.getWidth() / 640f;
        float scaleY = image.getHeight() / 640f;
        return PostProcessor.process(outputTensor, scaleX, scaleY);
    }


}
