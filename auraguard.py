import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


# COCO classes used by AuraGuard
TARGET_CLASSES = {
    0: "person",
    39: "bottle",
    41: "cup",
    67: "cell phone",
}

# UI colors in BGR for OpenCV drawing
COLORS = {
    "person": (0, 255, 0),
    "cell phone": (0, 0, 255),
    "cup": (255, 0, 0),
    "bottle": (255, 0, 0),
}

PHONE_DISTRACTION_SECONDS = 10.0
HYDRATION_REMINDER_SECONDS = 30.0
CONFIDENCE_THRESHOLD = 0.35


def load_model(model_name: str = "yolov8n.pt") -> YOLO:
    """Load and return the YOLOv8 Nano model."""
    return YOLO(model_name)


def detect_objects(model: YOLO, frame: np.ndarray, conf_threshold: float = CONFIDENCE_THRESHOLD) -> List[Dict]:
    """Run YOLO inference and return only relevant class detections."""
    results = model(frame, verbose=False)
    detections: List[Dict] = []

    for result in results:
        if result.boxes is None:
            continue

        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)

        for box, conf, cls_id in zip(boxes, confs, classes):
            if conf < conf_threshold:
                continue
            if cls_id not in TARGET_CLASSES:
                continue

            x1, y1, x2, y2 = box.astype(int)
            detections.append(
                {
                    "class_id": cls_id,
                    "label": TARGET_CLASSES[cls_id],
                    "confidence": float(conf),
                    "bbox": (x1, y1, x2, y2),
                }
            )

    return detections


def update_phone_timer(
    phone_present: bool,
    phone_start_time: Optional[float],
    current_time: float,
) -> Tuple[Optional[float], float, bool]:
    """Track continuous phone presence and distraction state."""
    if phone_present:
        if phone_start_time is None:
            phone_start_time = current_time
        phone_elapsed = current_time - phone_start_time
        phone_warning = phone_elapsed >= PHONE_DISTRACTION_SECONDS
    else:
        phone_start_time = None
        phone_elapsed = 0.0
        phone_warning = False

    return phone_start_time, phone_elapsed, phone_warning


def update_hydration_timer(
    drink_present: bool,
    last_drink_time: float,
    current_time: float,
) -> Tuple[float, float, bool]:
    """Track hydration absence duration and reminder state."""
    if drink_present:
        last_drink_time = current_time

    hydration_elapsed = current_time - last_drink_time
    hydration_warning = hydration_elapsed >= HYDRATION_REMINDER_SECONDS

    return last_drink_time, hydration_elapsed, hydration_warning


def draw_hud(
    frame: np.ndarray,
    detections: List[Dict],
    phone_elapsed: float,
    hydration_elapsed: float,
    phone_warning: bool,
    hydration_warning: bool,
    fps: float,
) -> np.ndarray:
    """Draw all overlays: bounding boxes, status text, timers, and warnings."""
    output = frame.copy()
    height, width = output.shape[:2]

    # Draw filtered detection boxes
    for det in detections:
        label = det["label"]
        x1, y1, x2, y2 = det["bbox"]
        conf = det["confidence"]
        color = COLORS.get(label, (255, 255, 255))

        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            output,
            f"{label} {conf:.2f}",
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    # Top-left HUD
    cv2.putText(output, "AuraGuard Status", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(
        output,
        f"Phone detected time: {phone_elapsed:5.1f}s",
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        output,
        f"Hydration timer: {hydration_elapsed:5.1f}s",
        (20, 92),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        output,
        f"FPS: {fps:4.1f}",
        (20, 119),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # Center warnings
    if phone_warning:
        warning_text = "\u26a0 PHONE DISTRACTION DETECTED"
        text_size, _ = cv2.getTextSize(warning_text, cv2.FONT_HERSHEY_SIMPLEX, 0.95, 3)
        text_x = (width - text_size[0]) // 2
        text_y = height // 2
        cv2.putText(output, warning_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 0, 255), 3, cv2.LINE_AA)

    if hydration_warning:
        reminder_text = "\U0001f4a7 HYDRATION REMINDER: Take a sip of water!"
        text_size, _ = cv2.getTextSize(reminder_text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
        text_x = (width - text_size[0]) // 2
        text_y = (height // 2) + 45
        cv2.putText(output, reminder_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2, cv2.LINE_AA)

    return output


def process_frame(
    model: YOLO,
    frame: np.ndarray,
    phone_start_time: Optional[float],
    last_drink_time: float,
    current_time: float,
    fps: float,
) -> Tuple[np.ndarray, Optional[float], float]:
    """Run full pipeline on a single frame and return rendered frame plus updated timers."""
    detections = detect_objects(model, frame, CONFIDENCE_THRESHOLD)

    phone_present = any(det["label"] == "cell phone" for det in detections)
    drink_present = any(det["label"] in {"cup", "bottle"} for det in detections)

    phone_start_time, phone_elapsed, phone_warning = update_phone_timer(phone_present, phone_start_time, current_time)
    last_drink_time, hydration_elapsed, hydration_warning = update_hydration_timer(drink_present, last_drink_time, current_time)

    rendered = draw_hud(
        frame,
        detections,
        phone_elapsed,
        hydration_elapsed,
        phone_warning,
        hydration_warning,
        fps,
    )

    return rendered, phone_start_time, last_drink_time


def main() -> None:
    model = load_model("yolov8n.pt")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Unable to open webcam (index 0).")

    phone_start_time: Optional[float] = None
    last_drink_time = time.time()

    prev_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            current_time = time.time()
            dt = max(current_time - prev_time, 1e-6)
            fps = 1.0 / dt
            prev_time = current_time

            output_frame, phone_start_time, last_drink_time = process_frame(
                model,
                frame,
                phone_start_time,
                last_drink_time,
                current_time,
                fps,
            )

            cv2.imshow("AuraGuard - Smart Workspace & Hydration Monitor", output_frame)

            # Quit safely with Q or q
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
