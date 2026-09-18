import json

from config import DOCUMENT_CACHE_PATH
from document_processing.file_extraction import files_extractor


def main():
    documents = files_extractor()

    DOCUMENT_CACHE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with DOCUMENT_CACHE_PATH.open(
        "w",
        encoding="utf-8",
    ) as cache_file:
        for document in documents:
            record = {
                "page_content": document.page_content,
                "metadata": document.metadata,
            }
            cache_file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    default=str,
                )
                + "\n"
            )

    print(
        f"Saved {len(documents)} documents "
        f"to {DOCUMENT_CACHE_PATH}"
    )


if __name__ == "__main__":
    main()
