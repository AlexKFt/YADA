from storage_management.files_management import TextSector


def get_sentence(text: str, pos: int, by_sentence: bool = True):
    if by_sentence:
        start = end = pos
        while start != 0 and text[start] != '.':
            start -= 1
        while end != len(text) and text[end] != '.':
            end += 1

        return text[start + 1: end]
    else:
        neighbourhood = 50
        start = pos - neighbourhood
        end = pos + neighbourhood
        if start < 0:
            start = 0
        if end >= len(text):
            end = len(text)


def get_texts(corpus, files_sectors, sub_start, sub_end):
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


def split_corpus_v2(corpus_of_files: list[TextSector], n: int, floating_window=False, floating_window_size=0):
    corpus_sectors = []
    for file in corpus_of_files:
        f_sector_number = len(file.text) // n
        for i in range(f_sector_number):
            sub_start, sub_end = i * n, (i + 1) * n
            sub_text = file.text[sub_start:sub_end]
            corpus_sectors.append(TextSector(file.filename, sub_text, sub_start, sub_end, len(corpus_sectors)))

    return corpus_sectors
