sudo modprobe -r v4l2loopback
sudo modprobe v4l2loopback video_nr=2 card_label="AndroidCam" exclusive_caps=1

./scrcpy/scrcpy --video-source=camera --camera-size=1280x720 --camera-facing=back --v4l2-sink=/dev/video2 --no-playback