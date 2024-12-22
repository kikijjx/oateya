import docx
import pdfplumber
import PyPDF2
import re

def find(text):
    text = text.replace('\n', '')
    text = text.replace('!', '.')
    text = text.replace('?', '.')
    text = text.lower()
    text = re.sub(r'(?<!\w)-|-(?!\w)|[^\w\s.-]', '', text)
    text = text.split(sep='.')

    words = []
    for i in text:
        i = i.strip()
        if i:
            words.append(i.split())
    words = sum(words, [])

    #print(words)
    print('кол-во слов:')
    print(len(words))
    print('кол-во уникальных слов')
    print(len(set(words)))

    char1 = []
    char2 = []
    for i in set(words):
        if len(i) == 1:
            char1.append(i)
        if len(i) == 2:
            char2.append(i)
    print('список слов с одной буквой:')
    print(char1)
    print('список слов с двумя буквами:')
    print(char2)

def find_docx(path):
    doc = docx.Document(path)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text
    find(text)

def find_pdfplumber(path):
    with pdfplumber.open(path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    find(text)

def find_pypdf2(path):
    with open(path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text()
    find(text)

docx_path = "C:\\Users\\nikita\\Desktop\\vscode\\lab2\\VESNA.docx"
pdf_path = "C:\\Users\\nikita\\Desktop\\vscode\\lab2\\VESNA.pdf"

print("DOCX:")
find_docx(docx_path)

print("pdfplumber:")
find_pdfplumber(pdf_path)

print("PyPDF2:")
find_pypdf2(pdf_path)
