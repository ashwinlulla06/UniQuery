from document_processing.pdf_processing import pdf_preprocessor
from document_processing.image_processing import image_processor
from document_processing.text_processing import text_processor
import glob
import os
from pathlib import Path
import cv2
from langchain_classic.schema import Document

files = glob.glob("knowledge-base/**/*")

def files_extractor():
    documents = []
    for file in files:
        file_name = os.path.split(file)[1]
        extension = Path(file_name).suffix.lower().lstrip(".")
        print(f'Processing {file_name}....')

        if extension == 'pdf': 
            file = Path(file)
            pdf_docs = pdf_preprocessor(file)
            for doc in pdf_docs:
                doc.metadata['file_type'] = extension
                doc.metadata['file_name'] = os.path.basename(file)
                doc.metadata['document_id'] = os.path.basename(file).split('-')[0]
                doc.metadata['folder'] = os.path.dirname(file).split('\\')[-1]

            documents.extend(pdf_docs)

        elif extension in ['jpg', 'png', 'jpeg']:
            file = Path(file)
            image = cv2.imread(file)
            extracted_text, records, quality = image_processor(image, file_name)
            image_doc = Document(
                page_content = extracted_text,
                metadata = {
                    'source': file,
                    'total_lines_of_text': quality.get('line_count', 0),
                    'file_type': extension,
                    'file_size': os.path.getsize(file),
                    'confidence_score': quality.get('mean_confidence', 0.0),
                    'fallback_required': quality.get('fallback_required', False)
                }
            )

            image_doc.metadata['file_type'] = extension
            image_doc.metadata['file_name'] = os.path.basename(file)
            image_doc.metadata['document_id'] = os.path.basename(file).split('-')[0]
            image_doc.metadata['folder'] = os.path.dirname(file).split('\\')[-1]

            documents.append(image_doc)


        elif extension in ['txt', 'md']:
            file = Path(file)
            text_docs = text_processor(file)

            for doc in text_docs:
                doc.metadata['file_type'] = extension
                doc.metadata['file_name'] = os.path.basename(file)
                doc.metadata['document_id'] = os.path.basename(file).split('-')[0]
                doc.metadata['folder'] = os.path.dirname(file).split('\\')[-1]
            documents.extend(text_docs)

        else:
            print(f'The document format - {extension} is not supported by this model')

        print(f'Processing {file_name} done')

    return documents