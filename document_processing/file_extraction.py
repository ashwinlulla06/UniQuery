from pathlib import Path

import cv2
from langchain_classic.schema import Document

from config import KNOWLEDGE_BASE_DIR, PROJECT_ROOT
from document_processing.image_processing import image_processor
from document_processing.pdf_processing import pdf_preprocessor
from document_processing.text_processing import text_processor


def add_common_metadata(document, file_path, extension):
    try:
        source = file_path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        source = file_path.as_posix()

    document.metadata.update(
        {
            "source": source,
            "file_type": extension,
            "file_name": file_path.name,
            "document_id": file_path.name.split("-")[0],
            "folder": file_path.parent.name,
        }
    )


def files_extractor():
    documents = []
    files = sorted(
        path
        for path in KNOWLEDGE_BASE_DIR.rglob("*")
        if path.is_file()
    )

    for file_path in files:
        file_name = file_path.name
        extension = file_path.suffix.lower().lstrip(".")
        print(f"Processing {file_name}....")

        if extension == "pdf":
            pdf_documents = pdf_preprocessor(file_path)

            for document in pdf_documents:
                add_common_metadata(
                    document,
                    file_path,
                    extension,
                )

            documents.extend(pdf_documents)

        elif extension in {"jpg", "png", "jpeg"}:
            image = cv2.imread(str(file_path))

            if image is None:
                raise ValueError(
                    f"Could not read image: {file_path}"
                )

            extracted_text, records, quality = image_processor(
                image,
                file_name,
            )

            image_document = Document(
                page_content=extracted_text,
                metadata={
                    "total_lines_of_text": quality.get(
                        "line_count",
                        0,
                    ),
                    "file_size": file_path.stat().st_size,
                    "confidence_score": quality.get(
                        "mean_confidence",
                        0.0,
                    ),
                    "fallback_required": quality.get(
                        "fallback_required",
                        False,
                    ),
                },
            )
            add_common_metadata(
                image_document,
                file_path,
                extension,
            )
            documents.append(image_document)

        elif extension in {"txt", "md"}:
            text_documents = text_processor(file_path)

            for document in text_documents:
                add_common_metadata(
                    document,
                    file_path,
                    extension,
                )

            documents.extend(text_documents)

        else:
            print(
                f"The document format '{extension}' "
                "is not supported."
            )

        print(f"Processing {file_name} done")

    return documents
