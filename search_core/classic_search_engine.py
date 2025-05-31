import math
from collections import defaultdict
import text_processing.query_processing as proc


class ClassicSearchEngine:
    def __init__(self, documents, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.docs = documents
        self.N = len(documents)
        self.avgdl = 0
        self.doc_lens = {}
        self.inverted_index = defaultdict(list)
        self.doc_freq = {}
        self._build_index()

    def _tokenize(self, text):
        return text.lower().split()

    def _build_index(self):
        total_len = 0
        for doc_id, text in self.docs.items():
            terms = self._tokenize(text)
            self.doc_lens[doc_id] = len(terms)
            total_len += len(terms)
            freqs = defaultdict(int)
            for term in terms:
                freqs[term] += 1
            for term, freq in freqs.items():
                self.inverted_index[term].append((doc_id, freq))
        self.avgdl = total_len / self.N
        for term, postings in self.inverted_index.items():
            self.doc_freq[term] = len(postings)

    def search(self, query, top_k=10, method="bm25"):
        terms = self._tokenize(query)
        term_set = set(terms)
        scores = defaultdict(float)
        matched_counts = defaultdict(int)

        # собираем терм-подсчёты по документам
        for term in term_set:
            if term not in self.inverted_index:
                continue
            idf = self._idf(term)
            for doc_id, freq in self.inverted_index[term]:
                doc_len = self.doc_lens[doc_id]
                if method == "bm25":
                    scores[doc_id] += self._bm25_term_score(freq, doc_len, idf)
                elif method == "tf-idf":
                    scores[doc_id] += self._tf_idf_score(freq, idf)
                matched_counts[doc_id] += 1

        return self.useWeakAnd(term_set, scores, matched_counts, top_k)

    def _idf(self, term):
        # классический BM25 idf
        df = self.doc_freq.get(term, 0)
        # добавляем 0.5 сгладку
        return math.log((self.N - df + 0.5) / (df + 0.5) + 1)

    def _bm25_term_score(self, freq, doc_len, idf):
        return idf * (freq * (self.k1 + 1) / (freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)))

    def _tf_idf_score(self, freq, idf):
        return freq * idf

    def useWeakAnd(self, term_set, scores, matched_counts, top_k):
        # применяем мягкую пенальтизацию
        final_scores = []
        for doc_id, score in scores.items():
            matched = matched_counts[doc_id]
            total = len(term_set)
            # множитель: доля найденных
            penalty = matched / total
            final_scores.append((doc_id, score * penalty, matched))

        # сортируем по скору
        final_scores.sort(key=lambda x: x[1], reverse=True)
        return final_scores[:top_k]


def straight_search(texts: list[str], substring: str):

    occurrences = [0 for i in range(len(texts))]
    for i in range(len(texts)):
        pos = texts[i].find(substring, 0)

        while pos != -1:
            occurrences[i] += 1
            pos = texts[i].find(substring, pos + 1)

    return occurrences


# Пример использования
if __name__ == "__main__":
    docs = {
        1: "The cat sits on the mat.",
        2: "A quick brown fox jumps over the lazy dog.",
        3: "Deep learning models are powerful tools for NLP tasks.",
        4: "Transformers use attention mechanisms to process sequences."
    }
    preparator = proc.TextPreparator()
    docs = {i: ' '.join(preparator.preprocess_text(docs[i])) for i in range(1, len(docs)+1)}
    engine = ClassicSearchEngine(docs)
    query = preparator.preprocess_text("How do transformers work in NLP?")
    results = engine.search(' '.join(query), top_k=2)
    print("Результаты поиска:")
    for doc_id, score, matched in results:
        print(f"Doc {doc_id}: score={score:.4f}, matched_terms={matched}")
