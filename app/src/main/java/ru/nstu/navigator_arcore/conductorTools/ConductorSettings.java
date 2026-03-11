package ru.nstu.navigator_arcore.conductorTools;

import com.google.gson.annotations.SerializedName;

import java.util.List;
import java.util.Map;

public class ConductorSettings {
    @SerializedName("Object_by_dangerous_levels")
    public Map<Integer, List<DetectedObjects>> objectDangerousLevels;
    @SerializedName("feedback_template_vibration")
    public Map<Integer, VibrationPatterns> feedbackTemplateVibration;
}