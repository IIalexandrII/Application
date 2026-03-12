package ru.nstu.navigator_arcore;

import android.app.Activity;
import android.content.Context;
import android.media.Image;
import android.os.VibrationEffect;
import android.os.Vibrator;
import android.speech.tts.TextToSpeech;
import android.speech.tts.Voice;
import android.util.DisplayMetrics;
import android.util.Log;
import android.widget.TextView;

import java.util.Comparator;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

import ru.nstu.navigator_arcore.conductorTools.*;
import ru.nstu.navigator_arcore.tools.BoundingBox;

public class Conductor implements TextToSpeech.OnInitListener {
    // поля компонент и настроек
    private Context context;
    private ConductorSettings settings;
    private TextToSpeech tts;

    // поля для определения направления
    private int wWidth, wHeight;
    private int gridCellWidth, gridCellHeight;

    // тайминги озвучки
    private long lastSpeechTime = 0;
    private static final long SPEECH_COOLDOWN = 5000;

    // вибрация по дистанции
    private DepthObstacleDetector depthObstacleDetector;

    //test----------------------
    TextView speakerText;

    public Conductor(Context context, ConductorSettings settings){
        this.context = context;
        this.settings = settings;

        DisplayMetrics dMetrics = new DisplayMetrics();
        ((Activity) this.context).getWindowManager().getDefaultDisplay().getMetrics(dMetrics);
        this.wHeight = dMetrics.heightPixels;
        this.wWidth  = dMetrics.widthPixels;

        this.tts = new TextToSpeech(this.context, this);

        this.gridCellHeight = this.wHeight / 3;
        this.gridCellWidth  = this.wWidth  / 3;

        this.speakerText = ((Activity) this.context).findViewById(R.id.testDirection);

        this.depthObstacleDetector = new DepthObstacleDetector(context);
    }

    public void notification(List<BoundingBox> bboxes, Image depthImage){
        this.notification(bboxes);
        depthObstacleDetector.processDepth(depthImage);
    }
    public void notification(List<BoundingBox> bboxes) {
        if (bboxes == null || bboxes.isEmpty()) return;

        sortByDistance(bboxes);

        // положение в сетке для направления
        StringBuilder TempText = new StringBuilder();
        StringBuilder speechBuilder = new StringBuilder();
        int count = 0;
        for(BoundingBox box: bboxes){
            String direction = getDirectionText(box);
            String description = getObjectDescription(box.clazz);
            String dist = (box.distanceMeters != null) ? String.format(" в %.1f метрах", box.distanceMeters) : "";

            String fullLine = direction + description + dist;
            TempText.append(fullLine).append("\n");

            if (count < 2) {
                speechBuilder.append(direction).append(description).append(". ");
                count++;
            }
        }


        ((Activity) this.context).runOnUiThread(()->{
            this.speakerText.setText(TempText.toString());
        });

        long currentTime = System.currentTimeMillis();

        if (currentTime - lastSpeechTime > SPEECH_COOLDOWN) {
            speak(speechBuilder.toString());
            lastSpeechTime = currentTime;
        }
    }

    @Override
    public void onInit(int status) {
        if (status == TextToSpeech.SUCCESS) {
            tts.setLanguage(new Locale("ru")); // Ставим русский язык

            Set<Voice> voices = tts.getVoices();
            for (Voice voice : voices) {
                if (voice.getLocale().getLanguage().equals("ru") && !voice.isNetworkConnectionRequired()) {
                    if (voice.getName().contains("ruc")) {
                        tts.setVoice(voice);
                        break;
                    }
                }
            }
            //-ruf- // мужской
            //-rue- // женский
            //-rud- // мужской (слишком роботизированно)
            //-dfc- // женский (дефолтный)
            //-ruc- // женский
        }
    }

    private void speak(String text) {
        if (tts != null) {
            tts.setPitch(1.0f);
            tts.setSpeechRate(1.5f);
            tts.speak(text, TextToSpeech.QUEUE_FLUSH, null, "NotificationID");
        }
    }

    //===================================================================================
    private String getDirectionText(BoundingBox box) {
        int column = (int) (box.rect.centerX() / gridCellWidth);
        int row    = (int) (box.rect.centerY() / gridCellHeight);
        String res = "";
        switch (column){
            case 0: res += "Слева "; break;
            case 1: res += "Спереди "; break;
            case 2: res += "Справа "; break;
        }
        switch (row){
            case 0: res += "сверху "; break;
            case 1: res += "прямо "; break;
            case 2: res += "снизу "; break;
        }
        return res;
    }

    private int getDangerLevel(int objectId) {
        for (Map.Entry<Integer, List<DetectedObjects>> entry : settings.objectDangerousLevels.entrySet()) {
            for (DetectedObjects obj : entry.getValue()) {
                if (obj.id == objectId) {
                    return entry.getKey();
                }
            }
        }
        return 3;
    }

    // возвращает строку без пробела в конце !
    private String getObjectDescription(int objectId) {
        for (Map.Entry<Integer, List<DetectedObjects>> entry : settings.objectDangerousLevels.entrySet()) {
            for (DetectedObjects obj : entry.getValue()) {
                if (obj.id == objectId) {
                    return obj.description;
                }
            }
        }
        return "неизвестно";
    }


    private static void sortByDistance(List<BoundingBox> bboxes) {
        if (bboxes == null || bboxes.size() <= 1) return;

        bboxes.sort(Comparator.comparing(
                box -> box.distanceMeters,
                Comparator.nullsLast(Float::compare)
        ));
    }
}