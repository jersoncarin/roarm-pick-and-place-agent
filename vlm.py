from __future__ import annotations

import time
import json
import os
import socket
from multiprocessing import shared_memory
from typing import Union, List, Optional
from dataclasses import dataclass
import signal
import numpy as np
import torch
from PIL import Image
from collections import defaultdict
from dotenv import load_dotenv
from ultralytics.models.sam import SAM3SemanticPredictor

load_dotenv()

MODEL_ID = "facebook/sam3"
SOCKET_PATH = "/tmp/vlm.sock"
HOLD_MS = 1500
CONFIDENCE_THRESHOLD = 0.30

hf_token = os.environ.get("HUGGINGFACE_TOKEN")


def configure_torch() -> str:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        return "cuda"
    return "cpu"


def _recv_json(conn: socket.socket) -> dict:
    chunks = []
    while True:
        data = conn.recv(4096)
        if not data:
            break
        chunks.append(data)
        if b"\n" in data:
            break

    raw = b"".join(chunks).split(b"\n", 1)[0]
    return json.loads(raw.decode("utf-8"))


def _send_json(conn: socket.socket, payload: dict) -> None:
    try:
        conn.sendall((json.dumps(payload) + "\n").encode("utf-8"))
    except BrokenPipeError:
        print("Client disconnected before response was sent")
    except OSError as e:
        print(f"Socket send failed: {e}")


class VLM:
    def __init__(self) -> None:
        self.device = configure_torch()

        overrides = dict(
            conf=CONFIDENCE_THRESHOLD,
            task="segment",
            mode="predict",
            model="sam3.pt",
            half=True,
            save=False,
            verbose=False,
            imgsz=644,
            device=0 if self.device == 'cuda' else 'cpu'
        )
        self.predictor = SAM3SemanticPredictor(overrides=overrides)

        dummy_image = np.zeros((3, 224, 224), dtype=np.uint8)
        self.predictor.set_image(Image.fromarray(
            dummy_image.transpose(1, 2, 0), mode="RGB"))
        self.predictor(text=["dummy"])
        print("VLM model loaded and ready")

    def get_bbox_from_targets(
        self,
        image: np.ndarray,
        obj_targets: Union[str, List[str]],
    ) -> dict[str, list[float]]:
        grouped = defaultdict(
            lambda: {"bboxes": [], "masks": [], "scores": []})

        if image.dtype != np.uint8:
            image = image.astype(np.uint8)

        pil_image = Image.fromarray(image, mode="RGB")

        if isinstance(obj_targets, str):
            obj_targets = [obj_targets]

        prompts = [s.lower().strip() for s in obj_targets if s.strip()]

        self.predictor.set_image(pil_image)
        results = self.predictor(text=prompts)

        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                continue

            masks = r.masks.data if r.masks is not None else None

            for i, (box, score, cls_id) in enumerate(zip(r.boxes.xyxy, r.boxes.conf, r.boxes.cls)):
                x1, y1, x2, y2 = box

                if x2 <= x1 or y2 <= y1:
                    continue

                prompt = prompts[int(cls_id.item())]

                grouped[prompt]["bboxes"].append(box.cpu().numpy().tolist())
                grouped[prompt]["scores"].append(float(score.item()))

                if masks is not None:
                    grouped[prompt]["masks"].append(
                        masks[i].cpu().numpy().tolist())

        _bboxes = {}
        for prompt, data in grouped.items():
            _bboxes[prompt] = (
                data["bboxes"],
                data["masks"],
                data["scores"],
            )

        return _bboxes


_running = True


def _handle_sigint(signum, frame):
    global _running
    _running = False


def serve() -> None:
    global _running

    signal.signal(signal.SIGINT, _handle_sigint)
    signal.signal(signal.SIGTERM, _handle_sigint)

    if os.path.exists(SOCKET_PATH):
        os.remove(SOCKET_PATH)
    print("Loading VLM model...")

    vlm = VLM()

    print("Running on device:", vlm.device)

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(SOCKET_PATH)
    server.listen(8)
    server.settimeout(1.0)

    print(f"VLM server listening on {SOCKET_PATH}")

    try:
        while _running:
            try:
                conn, _ = server.accept()
            except socket.timeout:
                continue
            except OSError:
                break

            with conn:
                conn.settimeout(3.0)

                try:
                    req = _recv_json(conn)

                    shm_name = req["shm_name"]
                    shape = tuple(req["shape"])
                    dtype = np.dtype(req["dtype"])
                    targets = req["targets"]

                    shm = shared_memory.SharedMemory(name=shm_name)
                    try:
                        image = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
                        image_copy = image.copy()
                    finally:
                        shm.close()

                    result = vlm.get_bbox_from_targets(image_copy, targets)
                    _send_json(conn, {"ok": True, "bboxes": result})

                except socket.timeout:
                    try:
                        _send_json(
                            conn, {"ok": False, "error": "Connection timed out"})
                    except Exception:
                        pass
                except Exception as e:
                    try:
                        _send_json(conn, {"ok": False, "error": str(e)})
                    except Exception:
                        pass

    finally:
        print("Shutting down VLM server...")
        server.close()

        if os.path.exists(SOCKET_PATH):
            os.remove(SOCKET_PATH)


@dataclass
class BBoxHoldConfig:
    hold_timeout_ms: int = 1000


@dataclass
class TrackState:
    bboxes: list[list[float]]
    masks: list
    scores: list[float]
    seen_ms: float


class BBoxHoldCache:
    def __init__(self, config: Optional[BBoxHoldConfig] = None):
        self.config = config or BBoxHoldConfig()
        self._cache: dict[str, TrackState] = {}

    def get_bbox_from_targets(
        self,
        image: np.ndarray,
        obj_targets: Union[str, List[str]],
    ) -> dict[str, tuple[list[list[float]], list, list[float]]]:
        if image.dtype != np.uint8:
            raise ValueError(f"Expected uint8 image, got {image.dtype}")

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                f"Expected image shape (H, W, 3), got {image.shape}")

        target_names = [obj_targets] if isinstance(
            obj_targets, str) else list(obj_targets)

        shm = shared_memory.SharedMemory(create=True, size=image.nbytes)

        try:
            shm_image = np.ndarray(
                image.shape, dtype=image.dtype, buffer=shm.buf)
            shm_image[:] = image

            req = {
                "shm_name": shm.name,
                "shape": list(image.shape),
                "dtype": str(image.dtype),
                "targets": target_names,
            }

            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                client.connect(SOCKET_PATH)
                client.sendall((json.dumps(req) + "\n").encode("utf-8"))

                chunks = []
                while True:
                    data = client.recv(4096)
                    if not data:
                        break
                    chunks.append(data)
                    if b"\n" in data:
                        break

            raw = b"".join(chunks).split(b"\n", 1)[0]
            resp = json.loads(raw.decode("utf-8"))

            if not resp.get("ok"):
                raise RuntimeError(resp.get("error", "Unknown server error"))

            now_ms = time.monotonic() * 1000.0

            current_items: dict[str, tuple[list[list[float]],
                                           list, list[float]]] = resp.get("bboxes", {})
            result: dict[str, tuple[list[list[float]], list, list[float]]] = {}

            for name in target_names:
                new_item = current_items.get(name)
                cached = self._cache.get(name)

                if new_item is not None:
                    bboxes, masks, scores = new_item

                    self._cache[name] = TrackState(
                        bboxes=bboxes,
                        masks=masks,
                        scores=scores,
                        seen_ms=now_ms,
                    )
                    result[name] = (bboxes, masks, scores)
                    continue

                if cached is not None and now_ms - cached.seen_ms <= self.config.hold_timeout_ms:
                    result[name] = (cached.bboxes, cached.masks, cached.scores)
                else:
                    self._cache.pop(name, None)

            self._prune_expired(now_ms)
            return result

        finally:
            shm.close()
            shm.unlink()

    def _prune_expired(self, now_ms: float) -> None:
        expired = [
            name
            for name, state in self._cache.items()
            if now_ms - state.seen_ms > self.config.hold_timeout_ms
        ]
        for name in expired:
            self._cache.pop(name, None)


bbox_cache = BBoxHoldCache(
    config=BBoxHoldConfig(hold_timeout_ms=800)
)


def get_bbox_from_targets(
    image: np.ndarray,
    obj_targets: Union[str, List[str]],
) -> dict[str, list[float]]:
    return bbox_cache.get_bbox_from_targets(image, obj_targets)


if __name__ == "__main__":
    serve()
