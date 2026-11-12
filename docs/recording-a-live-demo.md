# Recording a short concurrency demonstration

SimpleScreenRecorder is already installed. No installation is needed, and no
recording has been started automatically.

1. Open **SimpleScreenRecorder** from Applications.
2. Arrange the camera preview and answer terminal side by side. Keep the source
   video ready and paused. Use a source with a clearly visible person so the
   current gate can fire; frame the source tightly and hold the camera steady.
3. Select the screen or a rectangle containing the preview and terminal. Choose
   **15 FPS**, approximately **1280×720 or smaller**, with audio disabled.
4. Save as **Matroska (.mkv)** with **H.264**, using the **ultrafast** preset if
   available. Use a new dated name, for example
   `results/demo_concurrency_20261112_01.mkv`; never overwrite an earlier take.
5. Once alignment is ready, start recording. Press **S in the camera preview**,
   then start source playback. Capture 30–60 seconds with several answer updates.
6. Stop and save the recording. Play it back to confirm it shows moving preview,
   increasing Scout sequence numbers, and descriptions updating independently.

What to show:

- The camera preview keeps changing while an earlier answer remains on screen.
- A later answer appears while capture continues.
- The visible source age/stale label stays in the recording; leave incorrect
  descriptions visible rather than presenting a curated clip as accuracy evidence.
- Keep the matching result-folder name with the video so timing and RAM can be
  checked against the saved logs.

This is visual demonstration evidence. Screen capture consumes CPU/memory and may
change timing, so label it as a **screen-recorded demonstration**, separate from
clean performance measurements. If recording makes the preview sluggish, reduce
recording resolution or frame rate before changing model settings.

Existing trials contain analyzed stills and timing logs, not a full moving-screen
recording. A new recording is needed to show actual moving capture; a slideshow
of saved inputs must not be presented as the original live session.
