import os
import streamlit as st
import time

import search_core.classic_search_engine as search
import search_core.knowledge_base as kb

import storage_management.files_management as fm
import storage_management.utils as fm_utils

from text_processing.query_processing import TextPreparator
import text_processing.document_processing as dproc


def measure_time(func, *args, **kwargs):
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"Функция '{func.__qualname__}' выполнилась за {elapsed:.6f} секунд")
    return result


CORPUS_SECTOR_LENGTH = 512
TOP_K = 4
FILES_ROOT = r"C:\Users\Alexander\source\yada\Storage"


def cum_method_metric_for_texts(fun, key_words: list[str], texts: list[str]):
    texts_metric_values = [0 for i in range(len(texts))]

    for i in range(len(key_words)):
        metric = fun(texts, key_words[i])
        texts_metric_values = [texts_metric_values[j] + metric[j] for j in range(len(texts))]

    return sorted(enumerate(texts_metric_values), key=lambda x: x[1], reverse=True)


def read_uploaded_files(uploaded_files, reader):
    file_names = set()
    corpus_of_files = []

    for ufile in uploaded_files:
        file_names.add(ufile.name)
    app_state.name_to_path_dict = fm_utils.get_paths_from_names(file_names, FILES_ROOT)
    for name in app_state.name_to_path_dict.keys():
        path = app_state.name_to_path_dict[name]
        file_text = reader.read_file(path, True)
        print(f"filename: {ufile.name} len: {len(file_text)}")
        file = fm.TextSector(path, file_text, len(file_text))
        corpus_of_files.append(file)

    return corpus_of_files


def get_names_by_index(indexes, corpus_sectors):
    res = []
    unique = set()
    for idx in indexes:
        name = corpus_sectors[idx].filename
        if name not in unique:
            unique.add(name)
            res.append(name)
    return res


class GlobalState:
    def __init__(self):
        term_dict = fm_utils.parse_dict_json(r"C:\Users\Alexander\source\yada\Storage\oil_and_gas_dict.json")

        self.knowledge_base = kb.KnowledgeBase(term_dict=term_dict)
        self.reader = fm.Reader()
        self.preparator = TextPreparator()

        self.ready_for_file_load = True
        self.start_search = False
        self.show_res = False
        self.show_sidebar_res = False

        self.lookup_filename_end = False
        self.tf_idf_search_end = False
        self.bm25_search_end = False
        self.sem_search_end = False

        self.index_file = ""
        self.waiting_for_reading_index_file = False
        self.waiting_for_writing_index_file = False
        self.selected_file = ""
        self.name_to_path_dict = dict()
        self.current_index = ""
        self.current_index_descriptor = ""
        self.default_index = r"C:\Users\Alexander\source\yada\Storage\full_index.faiss"
        self.default_index_descriptor = r"C:\Users\Alexander\source\yada\Storage\full_index.msgpack"

        self.key_value = 2


def get_global_state():
    if "global_state" in st.session_state:
        return st.session_state["global_state"]
    else:
        st.session_state["global_state"] = GlobalState()
        return st.session_state["global_state"]


app_state = get_global_state()

st.title("🔍 Семантический поиск по файлам")

uploaded_files = st.file_uploader("Загрузите текстовый файл (.txt, .pdf, .docx)",
                                  type=["txt", "pdf", "docx"],
                                  accept_multiple_files=True)

if st.button("Проиндексировать выбранные файлы"):
    app_state.ready_for_file_load = True

if st.button("Сохранить полученный индекс в файл"):
    app_state.waiting_for_writing_index_file = True

app_state.current_index = st.text_input("Введите имя файла индекса для записи:", value=fm_utils.get_beauty_file_name(app_state.current_index))

# Описание процесса сохранения рабочего индекса на диск
if app_state.waiting_for_writing_index_file:
    if app_state.current_index:
        app_state.current_index_descriptor = os.path.join(FILES_ROOT, app_state.current_index.split(".")[0] + ".msgpack")
        app_state.knowledge_base.write_index_file(os.path.join(FILES_ROOT, app_state.current_index))
        fm.serialize_text_sectors(app_state.corpus_sectors, app_state.current_index_descriptor)
        st.write(f"Данные загружены в {app_state.current_index}!")
    else:
        app_state.knowledge_base.write_index_file(app_state.default_index)
        fm.serialize_text_sectors(app_state.corpus_sectors, app_state.default_index_descriptor)
        st.write(f"Данные загружены в {app_state.default_index}!")
    app_state.waiting_for_writing_index_file = False


if st.button("Загрузить существующий индекс"):
    app_state.waiting_for_reading_index_file = True
if app_state.waiting_for_reading_index_file:
    app_state.index_file = st.file_uploader("Выберите файл индекса(.faiss)", type=["faiss"],
                                            key=app_state.key_value)

# Описание процесса получение файлов из сохранённого индекса
if app_state.index_file:
    app_state.waiting_for_index_file = False
    app_state.key_value *= 2
    app_state.current_index, app_state.current_index_descriptor, app_state.corpus_sectors = (
        app_state.reader.read_index(app_state.index_file,
                                    app_state.knowledge_base,
                                    app_state.default_index,
                                    FILES_ROOT))
    app_state.file_names = list(
        set(app_state.corpus_sectors[i].filename for i in range(len(app_state.corpus_sectors))))
    for file in app_state.file_names:
        content = app_state.reader.read_file(file, isPath=True)
        for i in range(len(app_state.corpus_sectors)):
            if app_state.corpus_sectors[i].filename == file:
                app_state.corpus_sectors[i].text = content[
                                                   app_state.corpus_sectors[i].start: app_state.corpus_sectors[
                                                       i].end]

    app_state.texts = [tp.text for tp in app_state.corpus_sectors]
    app_state.bm25_engine = measure_time(search.ClassicSearchEngine,
                                         {i: app_state.texts[i] for i in range(len(app_state.texts))})

    if app_state.knowledge_base.index is not None:
        st.write(f"Успешно загружен хранящий информацию из {len(app_state.file_names)} документов!")
    app_state.index_file = ""


st.sidebar.markdown("Использовать:")
use_full_search = st.sidebar.checkbox("Использовать глубокий поик")
use_filename_search = st.sidebar.checkbox("Искать в именах файлов")
use_tf_idf = st.sidebar.checkbox("TF-IDF")
use_semantic_search = st.sidebar.checkbox("Semantic search model")
use_bm25 = st.sidebar.checkbox("BM25")

TOP_K = int(st.sidebar.text_input("Искомое количество записей:", value=4))

# Описание процесса индексации новых файлов с диска
if uploaded_files and app_state.ready_for_file_load:
    app_state.ready_for_file_load = False

    with st.spinner("📖 Чтение и индексирование файла..."):
        app_state.corpus_of_files = read_uploaded_files(uploaded_files, app_state.reader)
        app_state.file_names = list(set(app_state.corpus_of_files[i].filename for i in range(len(app_state.corpus_of_files))))

        app_state.texts = []
        app_state.corpus_sectors = measure_time(dproc.split_corpus_v2, app_state.corpus_of_files,
                                                CORPUS_SECTOR_LENGTH)

        app_state.texts = [tp.text for tp in app_state.corpus_sectors]
        texts_for_model = [app_state.preparator.preprocess_text(text) for text in app_state.texts]
        measure_time(app_state.knowledge_base.build_corpus, texts_for_model)
        app_state.bm25_engine = measure_time(search.ClassicSearchEngine,
                                             {i: app_state.texts[i] for i in range(len(app_state.texts))})

    st.success(f"Обработка завершена! Прочитано {len(app_state.file_names)} файлов.")

if st.button("Искать"):
    app_state.start_search = True
    app_state.lookup_filename_end = False
    app_state.sem_search_end = False
    app_state.tf_idf_search_end = False
    app_state.bm25_search_end = False

# Описании панели создания запроса
query = st.text_input("Введите поисковый запрос:")

if query and app_state.start_search:
    app_state.start_search = False

    app_state.key_words_lemms = app_state.preparator.get_keywords_rake(query, lemmatization=True)

    synonyms_extension = []
    for kew_word in app_state.key_words_lemms:
        if kew_word in app_state.knowledge_base.term_dict:
            synonyms_extension.extend(app_state.knowledge_base.term_dict[kew_word])
    app_state.key_words_lemms.extend(synonyms_extension)
    print(app_state.key_words_lemms)
    app_state.stemmed_words = app_state.preparator.preprocess_text(' '.join(app_state.key_words_lemms),
                                                                   stemming=True).split()

    if use_full_search:
        app_state.D, app_state.I = app_state.knowledge_base.search(' '.join(app_state.key_words_lemms), TOP_K)
        app_state.full_search_files = get_names_by_index(app_state.I[0], app_state.corpus_sectors)

        files_content = []

        for file in app_state.full_search_files:
            files_content.append(app_state.reader.read_file(file, True))
        app_state.bm25_engine = search.ClassicSearchEngine({i: files_content[i] for i in range(len(app_state.full_search_files))})
        app_state.full_search_bm25_res = app_state.bm25_engine.search(' '.join(app_state.key_words_lemms),
                                                                      top_k=len(app_state.full_search_files), method="bm25")

        app_state.final_full_search_res = []
        final_files_indices = []
        for idx, score, matched in app_state.full_search_bm25_res:
            if score > 0:
                app_state.final_full_search_res.append((idx, score, matched))
                final_files_indices.append(idx)
        app_state.full_search_files = [app_state.full_search_files[i] for i in range(len(app_state.full_search_files)) if i in final_files_indices]
        app_state.full_search_end = True
    else:
        if use_semantic_search:
            app_state.D, app_state.I = measure_time(app_state.knowledge_base.search, ' '.join(app_state.key_words_lemms),
                                                    TOP_K)
            app_state.sem_files = get_names_by_index(app_state.I[0], app_state.corpus_sectors) # возвращает укороченный список уникальных имён
            app_state.sem_search_end = True

        if use_filename_search:
            app_state.beauty_file_names = [fm_utils.get_beauty_file_name(file_name).lower() for file_name in app_state.file_names]
            app_state.file_names_lookup = measure_time(cum_method_metric_for_texts, search.straight_search,
                                                       app_state.stemmed_words, app_state.beauty_file_names)
            app_state.lookup_filename_end = True
        if use_tf_idf:
            app_state.tf_idf_res = measure_time(app_state.bm25_engine.search, ' '.join(app_state.key_words_lemms),
                                                top_k=TOP_K, method="tf-idf")
            app_state.tf_idf_files = get_names_by_index([i for i, s, m in app_state.tf_idf_res], app_state.corpus_sectors)

            app_state.tf_idf_search_end = True
        if use_bm25:
            app_state.bm_25_res = measure_time(app_state.bm25_engine.search, ' '.join(app_state.key_words_lemms),
                                               top_k=TOP_K, method="bm25")
            app_state.bm25_files = get_names_by_index([i for i, s, m in app_state.bm_25_res], app_state.corpus_sectors)
            app_state.bm25_search_end = True

    app_state.show_sidebar_res = True
    app_state.show_res = True

# Описание боковой панели
if app_state.show_sidebar_res:
    key_value = 3
    if use_full_search and app_state.full_search_end:
        st.sidebar.markdown("Найденные файлы:")
        for file in app_state.full_search_files:
            key_value *= 3
            if st.sidebar.button(file, key=key_value):
                app_state.selected_file = file
                app_state.show_res = False
    else:
        if use_filename_search and app_state.lookup_filename_end:
            st.sidebar.markdown("Файлы с подходящим названием:")
            k = 0
            for idx, score in app_state.file_names_lookup:
                if k > TOP_K - 1:
                    break
                k += 1
                key_value *= 3
                if st.sidebar.button(app_state.file_names[idx], key=key_value):
                    app_state.selected_file = app_state.file_names[idx]
                    app_state.show_res = False
        if use_tf_idf and app_state.tf_idf_search_end:
            st.sidebar.markdown("TF-IDF files:")
            for i in range(len(app_state.tf_idf_files) if len(app_state.tf_idf_files) < TOP_K else TOP_K):
                name = app_state.tf_idf_files[i]
                key_value *= 3
                if st.sidebar.button(name, key=key_value):
                    app_state.selected_file = name
                    app_state.show_res = False
        if use_semantic_search and app_state.sem_search_end:
            st.sidebar.markdown("Semantic search files:")
            for name in app_state.sem_files:
                key_value *= 3
                if st.sidebar.button(name, key=key_value):
                    app_state.selected_file = name
                    app_state.show_res = False
        if use_bm25 and app_state.bm25_search_end:
            st.sidebar.markdown("BM25 files:")
            for name in app_state.bm25_files:
                key_value *= 3
                if st.sidebar.button(name, key=key_value):
                    app_state.selected_file = name
                    app_state.show_res = False

# Описание секции вывода результатов
if app_state.show_res:
    if st.button("Скрыть результаты"):
        app_state.show_res = False

    st.subheader("🔎 Результаты поиска:")
    if use_full_search and app_state.full_search_end:
        for doc_id, score, matched in app_state.final_full_search_res:
            file = app_state.full_search_files[doc_id]
            for i, d in zip(app_state.I[0], app_state.D[0]):
                if app_state.corpus_sectors[i].filename == file:
                    st.markdown(f"*Ранг по семантическому поиску:{d}, оценка по BM25: {score}, найдено {matched} совпадений")
                    st.write(fm_utils.get_beauty_file_name(app_state.corpus_sectors[i].filename))
                    st.write(app_state.corpus_sectors[i].text)
                    st.markdown("---")
    else:
        if use_filename_search and app_state.lookup_filename_end:
            k = 0
            for idx, score in app_state.file_names_lookup:
                if k > TOP_K - 1:
                    break
                st.markdown(f"**Ключевые слова в названии файла:** {score}")
                name = app_state.beauty_file_names[idx]
                st.write(name)
                st.markdown("---")
                k += 1

        if use_tf_idf and app_state.tf_idf_search_end:
            for doc_id, score, matched in app_state.tf_idf_res:
                st.markdown(f"TF-IDF score={score:.4f}, matched_terms={matched}")
                st.write(fm_utils.get_beauty_file_name(app_state.corpus_sectors[doc_id].filename))
                st.write(app_state.corpus_sectors[doc_id].text)
                st.markdown("---")
        if use_semantic_search and app_state.sem_search_end:
            k = 1
            for idx, dist in zip(app_state.I[0], app_state.D[0]):
                st.markdown(f"**Sentence Transformer rang:** {k}")
                st.write(fm_utils.get_beauty_file_name(app_state.corpus_sectors[idx].filename))
                st.write(app_state.corpus_sectors[idx].text)
                st.markdown("---")
                k += 1

        if use_bm25 and app_state.bm25_search_end:
            for doc_id, score, matched in app_state.bm_25_res:
                st.markdown(f"BM25 score={score:.4f}, matched_terms={matched}")
                st.write(fm_utils.get_beauty_file_name(app_state.corpus_sectors[doc_id].filename))
                st.write(app_state.corpus_sectors[doc_id].text)
                st.markdown("---")

# Описание режима показа файла
if app_state.selected_file:
    if app_state.selected_file in app_state.file_names:
        name = app_state.selected_file
    else:
        if app_state.selected_file in app_state.name_to_path_dict:
            name = app_state.name_to_path_dict[app_state.selected_file]
        else:
            name = fm_utils.get_paths_from_names(set(app_state.selected_file), FILES_ROOT)[app_state.selected_file]
    beauty_name = fm_utils.get_beauty_file_name(app_state.selected_file)
    st.subheader(f"Файл: {beauty_name}")
    st.download_button(
        label="📥 Скачать",
        data=app_state.reader.read_file(name, True),
        file_name=f"{beauty_name}",
        mime="text/plain"
    )
    if st.button("Вернуться к результатам поиска"):
        app_state.selected_file = ""
        app_state.show_res = True

    text = app_state.reader.read_file(app_state.selected_file, True)
    print(f"filename: {app_state.selected_file} Text length: {len(text)}")
    text_sections_borders = []

    if use_full_search:
        for idx in sorted(app_state.I[0]):
            if app_state.corpus_sectors[idx].filename == app_state.selected_file:
                text_sections_borders.append((app_state.corpus_sectors[idx].start, app_state.corpus_sectors[idx].end))
        for start_pos, end_pos in text_sections_borders:
            k = 0
            delta = len("<div style='background-color: blue;'></div>")
            if start_pos < len(text) and end_pos <= len(text):
                offset = k * delta
                text = text[
                       :start_pos + offset] + f"<div style='background-color: blue;'>{text[start_pos + offset:end_pos + offset]}</div>" + text[
                                                                                                                                          end_pos + offset:]
                k += 1
        for word in app_state.stemmed_words:
            text = text.replace(word, f"<span style='background-color: brown;'>{word}</span>")

    elif use_filename_search:
        pass
    elif use_tf_idf:
        for word in app_state.stemmed_words:
            text = text.replace(word, f"<span style='background-color: blue;'>{word}</span>")
    elif use_bm25:
        for word in app_state.stemmed_words:
            text = text.replace(word, f"<span style='background-color: blue;'>{word}</span>")
    elif use_semantic_search:
        for idx in sorted(app_state.I[0]):
            if app_state.corpus_sectors[idx].filename == app_state.selected_file:
                text_sections_borders.append((app_state.corpus_sectors[idx].start, app_state.corpus_sectors[idx].end))
        for start_pos, end_pos in text_sections_borders:
            k = 0
            delta = len("<div style='background-color: blue;'></div>")
            if start_pos < len(text) and end_pos <= len(text):
                offset = k * delta
                text = text[:start_pos+offset] + f"<div style='background-color: blue;'>{text[start_pos+offset:end_pos+offset]}</div>" + text[end_pos+offset:]
                k += 1

    st.markdown(f"<div style='font-size: 16px; line-height: 1.6'>{text}</div>", unsafe_allow_html=True)
    # st.text_area("Содержимое:", value=app_state.reader.read_file(app_state.selected_file, True), height=500)
