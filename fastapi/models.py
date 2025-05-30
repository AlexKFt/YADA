class Document:

    def __init__(self, name, path):
        self.name = name
        self.path = path
        self.key_phrases = []
        self.occurrences = {}

    def __str__(self):
        return f"name: {self.name} \npath: {self.path} \nkey_phrases: {self.key_phrases} \noccurences: {self.occurrences}\n"

    def __repr__(self):
        return f"name: {self.name} \npath: {self.path} \nkey_phrases: {self.key_phrases} \noccurences: {self.occurrences}\n"

    def calculate_score(self):
        n = len(self.key_phrases)
        self.score = 0
        for i in range(n):
            self.score += (n - i) / (n + 1) * len(self.occurrences[i])


def rank_responses(responses, n_return = 5):
    responses.sort(key=lambda x: x.score, reverse=True)
    if n_return > len(responses):
        n_return = len(responses)
    return responses[0:n_return]