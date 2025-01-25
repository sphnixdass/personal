import fitz  # PyMuPDF

def segregate_into_paragraphs(text):
    # Split the text into paragraphs using double newlines as separators
    paragraphs = [para.strip() for para in text.split("\n\n") if para.strip()]
    return paragraphs

def pdf_to_html(pdf_path, html_path):
    doc = fitz.open(pdf_path)
    html_content = "<!DOCTYPE html>\n<html>\n<body>\n"
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        with open('temptext.txt', 'a') as file:
            file.write(text)
        paragraphs = segregate_into_paragraphs(text)
        for i, para in enumerate(paragraphs, start=1):
            html_content += f"<h2>Page {page_num + 1} </h2>\n<p>{para}</p>\n"

    html_content += "</body>\n</html>"
    
    with open(html_path, 'w') as html_file:
        html_file.write(html_content)

# Usage
pdf_path = "/home/dass/Documents/Python/DassReader/The Maze Runner .pdf"  # Path to the input PDF
html_path = "outputMazeRunner.html"  # Path to save the HTML file
pdf_to_html(pdf_path, html_path)
print(f"HTML saved to {html_path}")
