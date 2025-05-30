import os
from os.path import isdir, isfile, join
import fitz  # PyMuPDF
import docx
import tempfile
import msgpack
import json


search_result_path = "search_result_for_original_docs.txt"


def get_beauty_file_name(file_name):
    return file_name.split('\\')[-1]


class TextSector:
    def __init__(self, filename, text, start=0, end=0, index_id=-1):
        self.filename = filename
        self.text = text
        self.start = start
        self.end = start + len(text) if text else end
        self.index_id = index_id

    def to_dict(self):
        return {
            'filename': self.filename,
            'start': self.start,
            'end': self.end,
            'index_id': self.index_id
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            filename=data['filename'],
            text="",
            start=data['start'],
            end=data['end'],
            index_id=data['index_id']
        )


def serialize(file_sectors, file_path):
    dict_list = [sector.to_dict() for sector in file_sectors]
    packed = msgpack.packb(dict_list, use_bin_type=True)
    with open(file_path, 'wb') as f:
        f.write(packed)


def deserialize(file_path):
    with open(file_path, 'rb') as f:
        dict_list = msgpack.unpackb(f.read(), raw=False)
        return [TextSector.from_dict(d) for d in dict_list]


class Reader:
    def __init__(self):
        self.filename = ""
        self.file_handler = None
        self.content = ""
        self.texts = []

    def read_index(self, file, knowledge_base, default_index_name, root_dir):
        if not file.name.endswith(".faiss"):
            index_name = default_index_name
        index_name = file.name
        descriptor = index_name.split(".")[0] + ".msgpack"
        paths = get_paths_from_names({index_name, descriptor}, root_dir)

        knowledge_base.read_index_file(paths[index_name])
        return paths[index_name], paths[descriptor], deserialize(paths[descriptor])

    def read_file(self, file, isPath=False) -> str:
        if isPath:
            ext = file.split(".")[-1]
            if ext == "txt":
                self.content = self.read_txt(file, isPath)
            elif ext == "pdf":
                self.content = self.read_pdf(file, isPath)
            elif ext == "docx":
                self.content = self.read_dock(file, isPath)
        else:
            if file.name.endswith(".txt"):
                self.content = self.read_txt(file)
            elif file.name.endswith(".pdf"):
                self.content = self.read_pdf(file)
            elif file.name.endswith(".docx"):
                self.content = self.read_dock(file)

        return self.content

    def read_txt(self, file, isPath=False) -> str:
        if isPath:
            with open(file, 'r', encoding="utf-8") as f:
                return f.read()
        else:
            return file.read().decode("utf-8")

    def read_pdf(self, file, isPath=False) -> str:
        if isPath:
            doc = fitz.open(file)
            buf = [page.get_text() for page in doc]
            doc.close()
        else:
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(file.read())
                tmp_file_path = tmp_file.name

            doc = fitz.open(tmp_file_path)
            buf = [page.get_text() for page in doc]
            doc.close()
            os.unlink(tmp_file_path)
        return '\n'.join(buf)

    def read_dock(self, file, isPath=False) -> str:
        if isPath:
            doc = docx.Document(file)
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp_file:
                tmp_file.write(file.read())
                tmp_file_path = tmp_file.name
            doc = docx.Document(tmp_file_path)
            os.unlink(tmp_file_path)

        buf = [para.text for para in doc.paragraphs if para.text.strip() != ""]
        return '\n'.join(buf)

    def split_corpus(self, corpus: str, n, files_sectors, floating_window=False, floating_window_size=0):
        corpus_sectors = []
        equal_sectors_number = len(corpus) // n
        for i in range(equal_sectors_number):
            sub_start, sub_end = i * n, (i + 1) * n
            sub_text, files_names = self.get_texts(corpus, files_sectors, sub_start, sub_end)
            text_sector = TextSector(files_names, sub_text)
            corpus_sectors.append(text_sector)

        if len(corpus) % n > 0:
            sub_text, files_names = self.get_texts(corpus, files_sectors, equal_sectors_number * n, len(corpus))
            corpus_sectors.append(TextSector(files_names, sub_text))
        return corpus_sectors

    def get_texts(self, corpus, files_sectors, sub_start, sub_end):
        files_names = []
        sub_text = corpus[sub_start:sub_end]
        file_idx = 0
        while sub_start < sub_end and file_idx < len(files_sectors):
            file = files_sectors[file_idx]
            if file.start <= sub_start < file.end:
                files_names.append(file.filename)
                sub_start = file.end
            file_idx += 1

        return sub_text, files_names

    def split_corpus_v2(self, corpus_of_files: list[TextSector], n: int, floating_window=False, floating_window_size=0):
        corpus_sectors = []
        for file in corpus_of_files:
            f_sector_number = len(file.text) // n
            for i in range(f_sector_number):
                sub_start, sub_end = i * n, (i + 1) * n
                sub_text = file.text[sub_start:sub_end]
                corpus_sectors.append(TextSector(file.filename, sub_text, sub_start, sub_end, len(corpus_sectors)))
            # if len(file.text) % n > 0:
            #     sub_start, sub_end = f_sector_number * n, len(file.text)
            #     sub_text = file.text[sub_start:sub_end]
            #     corpus_sectors.append(TextSector(file.filename, sub_text, sub_start, sub_end, len(corpus_sectors)))
        return corpus_sectors


def clear_file_data(file_name: str):
    with open(file_name, 'w') as f:
        pass


def log_str(file_name: str, text: str):
    with open(file_name, 'a', encoding='utf-8') as f:
        f.write(text + '\n')


def log_list(file_name: str, texts: list[str]):
    with open(file_name, 'a', encoding='utf-8') as f:
        for i, text in enumerate(texts):
            f.write(f"Part №{i}\n{text}" + '\n')
            f.write("-----------------------------------------------------------------------")


def get_paths_from_names(file_names: set[str], search_directory: str):
    if not isdir(search_directory):
        return dict()
    paths = dict()
    search_files_rec(search_directory, file_names, paths)
    return paths


def search_files_rec(path, file_names: set[str], paths:dict[str]):
    if len(file_names) == 0:
        return
    content = [obj for obj in os.listdir(path)]
    for obj in content:
        cur_path = join(path, obj)
        if isfile(cur_path):
            cur_file_name = cur_path.split('\\')[-1]
            if cur_file_name in file_names:
                paths[cur_file_name] = cur_path
                file_names.discard(cur_file_name)
        else:
            search_files_rec(cur_path, file_names, paths)


def get_directory_files(path) -> list:
    if not isdir(path):
        return []
    files = []
    get_directory_files_rec(path, files)
    return files


def get_directory_files_rec(path, files):
    content = [obj for obj in os.listdir(path)]
    for obj in content:
        cur_path = join(path, obj)
        if isfile(cur_path):
            files.append(cur_path)
        else:
            get_directory_files_rec(cur_path, files)


def get_lines_from_file(path) -> list[str]:
    lines = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            lines.append(line)
    return lines


def parse_dict(file_path, preparator):
    d = dict()
    lines = get_lines_from_file(file_path)
    for line in lines:
        split = line.split('-', 1)
        if len(split) != 2:
            continue
        word, definition = split
        d[word.lower()] = preparator.get_keywords_rake(definition)
    return d


def parse_dict_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = f.read()
    return json.loads(data)
