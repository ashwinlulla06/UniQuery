from pathlib import Path

import cv2
import numpy as np
from paddleocr import PaddleOCR

from config import PROOF_OF_OCR_DIR


ocr = PaddleOCR(
    lang="en",
    device="cpu",
    enable_mkldnn=False,
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=True,
    text_rec_score_thresh=0.0,
)


def image_processor(image_bgr, file_name):
    results = list(ocr.predict(image_bgr))
    records = []

    for result in results:
        payload = result.json

        if callable(payload):
            payload = payload()

        data = payload.get("res", payload)

        texts = list(data.get("rec_texts", []))
        scores = list(data.get("rec_scores", []))
        polygons = list(data.get("rec_polys", []))
        boxes = list(data.get("rec_boxes", []))

        for text, score, polygon, box in zip(
            texts,
            scores,
            polygons,
            boxes,
        ):
            text = text.strip()

            if not text:
                continue

            records.append(
                {
                    "text": text,
                    "confidence": float(score),
                    "polygon": np.asarray(
                        polygon
                    ).astype(int).tolist(),
                    "box": np.asarray(
                        box
                    ).astype(int).tolist(),
                }
            )

    extracted_text = "\n".join(
        record["text"] for record in records
    )
    scores = [
        record["confidence"] for record in records
    ]

    if scores:
        mean_confidence = float(np.mean(scores))
        minimum_confidence = float(np.min(scores))
        low_confidence_count = sum(
            score < 0.80 for score in scores
        )
        low_confidence_ratio = (
            low_confidence_count / len(scores)
        )
    else:
        mean_confidence = 0.0
        minimum_confidence = 0.0
        low_confidence_count = 0
        low_confidence_ratio = 1.0

    fallback_required = (
        not records
        or mean_confidence < 0.90
        or low_confidence_ratio > 0.15
    )

    quality = {
        "line_count": len(records),
        "mean_confidence": mean_confidence,
        "minimum_confidence": minimum_confidence,
        "low_confidence_count": low_confidence_count,
        "low_confidence_ratio": low_confidence_ratio,
        "fallback_required": fallback_required,
    }

    PROOF_OF_OCR_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    safe_file_name = Path(file_name).name

    visualize_ocr(
        image_bgr=image_bgr,
        records=records,
        output_path=(
            PROOF_OF_OCR_DIR
            / f"{safe_file_name}.png"
        ),
    )

    return extracted_text, records, quality


def visualize_ocr(image_bgr, records, output_path):
    output = image_bgr.copy()

    for record in records:
        box = np.asarray(
            record["box"],
            dtype=np.int32,
        )

        x1, y1, x2, y2 = map(
            int,
            box.tolist(),
        )

        score = float(record["confidence"])

        if score >= 0.90:
            color = (0, 255, 0)
        elif score >= 0.70:
            color = (0, 165, 255)
        else:
            color = (0, 0, 255)

        cv2.rectangle(
            img=output,
            pt1=(x1, y1),
            pt2=(x2, y2),
            color=color,
            thickness=2,
        )

        cv2.putText(
            img=output,
            text=f"{score:.2f}",
            org=(x1, max(y1 - 5, 15)),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
            color=color,
            thickness=2,
        )

    saved = cv2.imwrite(
        str(output_path),
        output,
    )

    if not saved:
        print(
            "Warning: annotated image could not "
            f"be saved to {output_path}"
        )

    return output
