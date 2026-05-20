# RoArm Pick and Place

This project controls a RoArm robotic arm with camera-based pick and place.

Use `gui.py` as the main app.

`main.py` is not used anymore and was removed to avoid confusion.

## File Overview

- `gui.py` runs the desktop app. This is where you start the project, view the camera feed, chat with the assistant, and control the workflow.
- `vlm.py` runs the vision model server. It loads SAM3 and returns detections to the GUI through `/tmp/vlm.sock`.
- `vision.py` handles camera access, ArUco detection, drawing overlays, and mask anchor point logic.
- `roarm.py` talks to the robotic arm over serial and sends pose commands.
- `pid.py` contains the PID controllers used for movement correction.
- `llm_tools.py` connects to an OpenAI-compatible API such as Ollama, OpenRouter, or OpenAI.
- `cam.sh` sets up an Android phone camera as a virtual Linux camera using `scrcpy` and `v4l2loopback`.
- `settings.json` stores GUI settings like camera index and serial port.
- `.env` stores secrets and runtime config.
- `.env.example` is the template for `.env`.

## Requirements

- Python 3.10 or newer
- Linux
- a USB camera, or an Android phone camera through `scrcpy`
- access to the RoArm serial device
- optional CUDA GPU if you want faster SAM3 inference

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## Environment Variables

Edit `.env` and set your values:

```env
HUGGINGFACE_TOKEN=your_token_here
OPENAI_BASE_URL=http://localhost:11434/v1
OPENAI_API_KEY=ollama
OPENAI_MODEL=qwen3-8b
```

## SAM3 and Hugging Face

This project uses `facebook/sam3`.

The model file `sam3.pt` should be in the project root folder.

You need to:

1. Make a Hugging Face account.
2. Request access to `facebook/sam3`.
3. Create a Hugging Face token.
4. Put that token in `.env` as `HUGGINGFACE_TOKEN`.

If Hugging Face access is not approved yet, SAM3 may fail to load.

## Android Phone Camera

If you want to use your Android phone instead of a USB camera, use `cam.sh`.

What it does:

- unloads and reloads `v4l2loopback`
- creates `/dev/video2` with the label `AndroidCam`
- starts `./scrcpy/scrcpy` with the phone camera as the video source
- forwards that stream into `/dev/video2`

Run it like this:

```bash
bash cam.sh
```

Then set the camera index in the GUI to `2` if needed, because `cam.sh` currently creates `/dev/video2`.

This requires:

- an Android phone connected and supported by `scrcpy`
- `v4l2loopback` installed
- permission to run the `sudo modprobe` commands in the script

## Run

Start the VLM server first:

```bash
python vlm.py
```

Then open another terminal, activate the same virtual environment, and run:

```bash
python gui.py
```

That is the normal way to run this project.

## Notes

- `gui.py` depends on `vlm.py`, so if the GUI opens but detection does not work, check that `vlm.py` is still running.
- The socket used between them is `/tmp/vlm.sock`.
- `sam3.pt` is in the project root folder and is used by `vlm.py`.
- Camera and serial settings can be changed from the GUI and are saved in `settings.json`.
- Default serial port is usually `/dev/ttyUSB0`.
- If you use `cam.sh`, the virtual camera is created at `/dev/video2`.

## Troubleshooting

- Camera issue: check the selected camera index and make sure your user can access `/dev/video*`.
- Android phone camera issue: make sure `cam.sh` is running, `scrcpy` can see the phone, and `/dev/video2` was created.
- Serial issue: check the selected serial device like `/dev/ttyUSB0` or `/dev/ttyACM0`.
- LLM issue: check `OPENAI_BASE_URL`, `OPENAI_API_KEY`, and `OPENAI_MODEL`.
- SAM3 issue: check `HUGGINGFACE_TOKEN`, make sure your Hugging Face access request was approved, and make sure `sam3.pt` is in the project root folder.
