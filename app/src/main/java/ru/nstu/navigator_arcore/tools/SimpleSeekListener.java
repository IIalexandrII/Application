package ru.nstu.navigator_arcore.tools;

import android.widget.SeekBar;

public class SimpleSeekListener implements SeekBar.OnSeekBarChangeListener  {
    public interface OnChange {
        void onChanged(int value);
    }

    public final OnChange listener;

    public SimpleSeekListener(OnChange listener) {
        this.listener = listener;
    }

    @Override public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
        listener.onChanged(progress);
    }

    @Override public void onStartTrackingTouch(SeekBar seekBar) {}
    @Override public void onStopTrackingTouch(SeekBar seekBar) {}
}
