package ru.nstu.navigator;

import ru.nstu.navigator.tools.OverlayView;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.util.Log;
import android.util.Size;
import android.widget.TextView;

import androidx.activity.EdgeToEdge;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.Camera;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;
import androidx.lifecycle.LifecycleOwner;

import com.google.common.util.concurrent.ListenableFuture;

import org.jspecify.annotations.NonNull;

import java.util.List;
import java.util.concurrent.CancellationException;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;

import ru.nstu.navigator.tools.BoundingBox;

public class MainActivity extends AppCompatActivity {

    private final int REQUEST_PERMISSION_CODE = 102030;
    private final String[] REQUIRED_PERMISSIONS = new String[] {"android.permission.CAMERA"};
    private ListenableFuture<ProcessCameraProvider> cameraProviderFuture;

    PreviewView preview;
    TextView resultTextView;
    OverlayView overlay;

    Model model;

    Executor executor = Executors.newSingleThreadExecutor();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        EdgeToEdge.enable(this);
        setContentView(R.layout.activity_main);

        overlay = findViewById(R.id.overlay);
        preview = findViewById(R.id.cameraView);
        resultTextView = findViewById(R.id.textResult);

        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main), (v, insets) -> {
            Insets systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars());
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom);
            return insets;
        });

        if(!this._checkPermission()){
            ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, REQUEST_PERMISSION_CODE);
        }

        cameraProviderFuture = ProcessCameraProvider.getInstance(this);
        cameraProviderFuture.addListener(() ->{
            try {
                ProcessCameraProvider cameraProvider = cameraProviderFuture.get();
                model = new Model("best.torchscript",this);
                startCamera(cameraProvider);
            } catch (Exception e) {
               Log.e("onCreate", "class MainActivity (onCreate): "+e.getMessage());
            }
        }, ContextCompat.getMainExecutor(this));
    }

    private boolean _checkPermission(){
        for(String perm : REQUIRED_PERMISSIONS){
            if(ContextCompat.checkSelfPermission(this, perm) != PackageManager.PERMISSION_GRANTED) return false;
        }
        return true;
    }

    void startCamera(@NonNull ProcessCameraProvider cameraProvider){
        Preview preview = new Preview.Builder().build();
        CameraSelector cameraSelector = new CameraSelector.Builder().requireLensFacing(CameraSelector.LENS_FACING_BACK).build();
        preview.setSurfaceProvider(this.preview.getSurfaceProvider());

        ImageAnalysis imageAnalysis = new ImageAnalysis.Builder()
                .setTargetResolution(new Size(640,640))
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build();


        imageAnalysis.setAnalyzer(executor, new ImageAnalysis.Analyzer() {
            @Override
            public void analyze(@androidx.annotation.NonNull ImageProxy image) {
                int rotation = image.getImageInfo().getRotationDegrees();
                long startTime = System.nanoTime();

                List<BoundingBox> boxes  = model.analyzeImage(image,rotation);

                long endTime = System.nanoTime();
                runOnUiThread(() -> {
                    overlay.setResults(boxes, model.Classes);

                    if (!boxes.isEmpty()) {
                        BoundingBox best = boxes.get(0);
                        resultTextView.setText(
                                model.Classes[best.clazz] + String.format(" %.2f", best.score) +
                                " Timing: " + (endTime - startTime) / 1_000_000+ " MS"
                        );
                    } else {
                        resultTextView.setText("—");
                    }
                });
                image.close();
            }
        });

        Camera camera = cameraProvider.bindToLifecycle((LifecycleOwner) this, cameraSelector, preview, imageAnalysis);

    }
}