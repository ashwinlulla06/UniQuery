import cv2
import numpy as np
import pymupdf

from langchain_community.document_loaders import PyPDFLoader
from document_processing.image_processing import image_processor
import os


def native_text_is_usable(text, minimum_characters=100):

    normalized = "".join(text.split())

    if len(normalized) < minimum_characters or not text:
        return False

    # Unicode replacement character indicates extraction problems.
    if "\ufffd" in text:
        return False

    alphanumeric_count = 0
    for character in normalized:
        if character.isalnum():   
            # Returns true for letters and numbers
            alphanumeric_count += 1


    alphanumeric_ratio = alphanumeric_count / len(normalized)

    # Helps reject text containing mostly corrupted symbols.
    if alphanumeric_ratio < 0.50:
        return False
    
    return True


def pymupdf_page_to_image(pdf_page, dpi=200):

    scale = dpi / 72   # Render this page about 2.78 times its normal PDF resolution.
    matrix = pymupdf.Matrix(scale, scale)

    pixmap = pdf_page.get_pixmap(    # Pixmap is the representation of the pixels of the image
        matrix=matrix,
        alpha=False,
    )

    image = np.frombuffer(
        pixmap.samples,
        dtype=np.uint8,
    ).reshape(
        pixmap.height,
        pixmap.width,
        pixmap.n,
    )  # height × width × channels

    # pixmap.n tells how many channels does the image has
    if pixmap.n == 1:
        image_bgr = cv2.cvtColor(
            image,
            cv2.COLOR_GRAY2BGR,    # converting gray-scale to bgr
        )
    elif pixmap.n == 3:
        image_bgr = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2BGR,      # converting rgb to bgr
        )
    elif pixmap.n == 4:
        image_bgr = cv2.cvtColor(
            image,
            cv2.COLOR_RGBA2BGR,     # converting rgba to bgr
        )
    else:
        raise ValueError(
            f"Unsupported channel count: {pixmap.n}"
        )

    return image_bgr

def pdf_preprocessor(file_path, minimum_native_characters=100, dpi=200):

    file_name = str(os.path.split(file_path)[1])
    loader = PyPDFLoader(file_path=str(file_path))
    page_documents = loader.load()

    with pymupdf.open(str(file_path)) as pdf:

        for page_index, page_document in enumerate(page_documents):

            native_text = page_document.page_content.strip()

            page_document.metadata["page_number"] = (page_index + 1)

            page_file_name = (
                f"{file_name} - "
                f"{page_document.metadata['page_number']}"
            )

            page_document.metadata["native_character_count"] = len(native_text)

            if native_text_is_usable(native_text, minimum_native_characters):

                page_document.page_content = native_text

                page_document.metadata["extraction_method"] = "native_pdf"

                page_document.metadata["ocr_used"] = False

                continue

            # Native extraction failed or returned too little text.
            pdf_page = pdf[page_index]

            image_bgr = pymupdf_page_to_image(pdf_page, dpi=dpi)

            ocr_text, records, quality = image_processor(image_bgr, page_file_name)

            ocr_text = (ocr_text or "").strip()

            if ocr_text:
                page_document.page_content = ocr_text

                page_document.metadata["extraction_method"] = "paddleocr"

            else:
                # Preserve partial native text if OCR completely fails.
                page_document.page_content = native_text

                page_document.metadata["extraction_method"] = "native_pdf_ocr_failed"

            page_document.metadata["ocr_used"] = True

            page_document.metadata["ocr_line_count"] = quality.get("line_count", 0)

            page_document.metadata["ocr_mean_confidence"] = quality.get("mean_confidence", 0.0)

            page_document.metadata["ocr_minimum_confidence"] = quality.get("minimum_confidence", 0.0)

            page_document.metadata["ocr_low_confidence_ratio"] = quality.get("low_confidence_ratio", 1.0)

            page_document.metadata["requires_vl_fallback"] = quality.get("fallback_required", False)

    return page_documents