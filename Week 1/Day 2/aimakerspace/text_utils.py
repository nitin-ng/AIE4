import os
import csv
from typing import List
import PyPDF2

class TextFileLoader:
    def __init__(self, path: str, encoding: str = "utf-8"):
        self.documents = []
        self.path = path
        self.encoding = encoding

    def load(self):
        if os.path.isdir(self.path):
            self.load_directory()
        elif os.path.isfile(self.path):
            if self.path.endswith(".txt"):
                self.load_txt_file()
            elif self.path.endswith(".pdf"):
                self.load_pdf_file()
            elif self.path.endswith(".csv"):
                self.load_csv_file()
            else:
                raise ValueError("Unsupported file type.")
        else:
            raise ValueError(
                "Provided path is neither a valid directory nor a supported file type."
            )

    def load_txt_file(self):
        with open(self.path, "r", encoding=self.encoding) as f:
            self.documents.append(f.read())

    def load_pdf_file(self):
        with open(self.path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            self.documents.append(text)

    def load_csv_file(self):
        with open(self.path, "r", encoding=self.encoding) as f:
            reader = csv.reader(f)
            text = ""
            for row in reader:
                text += " ".join(row) + "\n"  # Join columns with space and add a newline per row
            self.documents.append(text)

    def load_directory(self):
        for root, _, files in os.walk(self.path):
            for file in files:
                file_path = os.path.join(root, file)
                if file.endswith(".txt"):
                    self.load_txt_file_from_path(file_path)
                elif file.endswith(".pdf"):
                    self.load_pdf_file_from_path(file_path)
                elif file.endswith(".csv"):
                    self.load_csv_file_from_path(file_path)

    def load_txt_file_from_path(self, file_path):
        with open(file_path, "r", encoding=self.encoding) as f:
            self.documents.append(f.read())

    def load_pdf_file_from_path(self, file_path):
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            self.documents.append(text)

    def load_csv_file_from_path(self, file_path):
        with open(file_path, "r", encoding=self.encoding) as f:
            reader = csv.reader(f)
            text = ""
            for row in reader:
                text += " ".join(row) + "\n"
            self.documents.append(text)

    def load_documents(self):
        self.load()
        return self.documents


class CharacterTextSplitter:
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        assert (
            chunk_size > chunk_overlap
        ), "Chunk size must be greater than chunk overlap"

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, text: str) -> List[str]:
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunks.append(text[i : i + self.chunk_size])
        return chunks

    def split_texts(self, texts: List[str]) -> List[str]:
        chunks = []
        for text in texts:
            chunks.extend(self.split(text))
        return chunks


if __name__ == "__main__":
    # Example usage with a text file, a PDF file, or a CSV file.
    loader = TextFileLoader("data/sample.pdf")  # Replace with your file path
    loader.load()
    splitter = CharacterTextSplitter()
    chunks = splitter.split_texts(loader.documents)
    print(len(chunks))
    print(chunks[0])
    print("--------")
    print(chunks[1])
    print("--------")
    print(chunks[-2])
    print("--------")
    print(chunks[-1])
