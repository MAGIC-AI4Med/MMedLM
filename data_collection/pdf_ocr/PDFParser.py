'''
This file is mainly used for parsing PDFs. It's divided into text-based and image-based sections. 
For the text-based sections, PyPDF2 is used for direct reading. 
For the image-based sections, PaddleOCR is employed for recognition.
'''
import fitz
from paddleocr import PaddleOCR
from utils import find_tuples_recursive
import logging
logging.getLogger('ppocr').setLevel(logging.ERROR)


class PDFParser():
    def __init__(self,):
        # we need to declare the paddleOCR in the loop since the ISSUE of paddleOCR
        # self.paddleOCR = PaddleOCR(use_angle_cls=True, lang='ch')
        pass

    def extract_pdf(self, pdf_path):
        '''
            Determine the category of the PDF and extract the text to convert it into a TXT file.
        '''
        pdf_doc = fitz.open(pdf_path)
        if self.is_image_based_pdf(pdf_doc):
            text_ls = self.extract_image_based_pdf(pdf_path)
        else:
            text_ls = self.extract_text_based_pdf(pdf_doc)
        return text_ls

    def extract_text_based_pdf(self, pdf_doc):
        '''
            extract text-based pdf and return the text
        '''
        text_ls = []
        doc_pageCount = pdf_doc.pageCount

        for page_num in range(doc_pageCount):
            if (doc_pageCount < 100) or (page_num > doc_pageCount*0.018+3 and page_num < doc_pageCount-5):
                page = pdf_doc.loadPage(page_num)
                text_ls.append(page.getText())

        return text_ls

    def extract_image_based_pdf(self, pdf_path):
        '''
            extract image-based pdf and return the text
        '''
        
        # We have to declare the paddleOCR in the loop since the ISSUE of paddleOCR
        paddleOCR = PaddleOCR(use_angle_cls=True, lang='ru') # Russian
        result = paddleOCR.ocr(pdf_path, cls=True)
        text_ls = []
        for page_res in result:
            text = ''
            tuples_result = find_tuples_recursive(page_res)
            for item in tuples_result:
                if isinstance(item, tuple):
                    text += item[0] + ' '
            text_ls.append(text)
        return text_ls

    def is_image_based_pdf(self, pdf_doc):
        '''
            Determine whether the pdf is image-based or text-based
        '''
        text = ""
        page_total_num = pdf_doc.pageCount
        for page in pdf_doc:
            text += page.getText()

        alphanumeric_chars = sum(c.isalnum() for c in text)
        total_chars = len(text)+1
        alphanumeric_ratio = alphanumeric_chars / total_chars

        if alphanumeric_ratio < 0.5 or alphanumeric_chars < 200*page_total_num:
            return True  # Likely a image-based PDF
        else:
            return False  # Likely a text-based PDF

if __name__ == "__main__":
    pdf_parser = PDFParser()
    pdf_path = "demo_data/text_based.pdf"
    text = pdf_parser.extract_pdf(pdf_path)
    print(text)
