from rake_nltk import Rake
import pymorphy3
import re
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

# nltk.download('stopwords')
class TextPreparator:
    def __init__(self, text="", language="russian"):
        self.language = language
        self.morph = pymorphy3.MorphAnalyzer()
        self.stemmer = SnowballStemmer(self.language)
        self.rake = Rake(language=f"{self.language}", max_length=2)
        self.text = text
        self.russian_stopwords = set(stopwords.words(self.language))

    def preprocess_text(self, text: str,
                        lemmatization: bool=False,
                        stop_words_del=True,
                        stemming=False) -> str:

        words = re.sub(r'[^\w\s]', ' ', text.lower()).split()

        if stop_words_del:
            words = [word for word in words if word not in self.russian_stopwords]
        if lemmatization:
            words = [self.morph.parse(word)[0].normal_form for word in words]
        if stemming:
            words = [self.stemmer.stem(word) for word in words]

        return ' '.join(words)

    def get_keywords_rake(self, text: str,
                          lemmatization: bool = False,
                          stop_words_del=True,
                          stemming=False):
        self.rake.extract_keywords_from_text(text)
        frases = self.rake.get_ranked_phrases()
        self.text = ""
        for frase in frases:
            self.text += self.preprocess_text(frase, lemmatization=lemmatization, stop_words_del=stop_words_del,
                                             stemming=stemming) + " "
        return self.text.split()
