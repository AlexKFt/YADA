import json
import os
from os.path import isdir, isfile, join


def get_beauty_file_name(file_name):
    return file_name.split('\\')[-1]


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


def get_dir_files(search_directory: str, filter=lambda x: x):
    if not isdir(search_directory):
        return []
    index_list = []
    get_dir_files_rec(search_directory, index_list, filter)
    return index_list


def get_dir_files_rec(path, index_list, filter):
    content = [obj for obj in os.listdir(path)]
    for obj in content:
        cur_path = join(path, obj)
        if isfile(cur_path):
            cur_file_name = cur_path.split('\\')[-1]
            if filter(cur_file_name):
                index_list.append((cur_file_name, cur_path))

def parse_dict_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = f.read()
    return json.loads(data)


def parse_dashed_dict(file_path, preparator):
    d = dict()
    lines = get_lines_from_file(file_path)
    for line in lines:
        split = line.split('-', 1)
        if len(split) != 2:
            continue
        word, definition = split
        d[word.lower()] = preparator.get_keywords_rake(definition)
    return d


def get_lines_from_file(path) -> list[str]:
    lines = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            lines.append(line)
    return lines