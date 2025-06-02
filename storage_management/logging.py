

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