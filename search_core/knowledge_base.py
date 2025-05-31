import math
import torch
from transformers import AutoTokenizer, AutoModel
import faiss


class KnowledgeBase:
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2', device=None, term_dict=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.term_dict = term_dict if term_dict else dict()
        self.corpus_embeddings = None
        self.corpus_texts = None
        self.index = None
        faiss.omp_set_num_threads(8)

    def build_corpus(self, corpus_texts):
        nlist = int(math.sqrt(len(corpus_texts)))

        self.corpus_texts = corpus_texts
        self.corpus_embeddings = self.encode(corpus_texts)
        emb_np = self.corpus_embeddings.numpy().astype('float32')
        dim = emb_np.shape[1]
        quantizer = faiss.IndexFlatL2(dim)  # dim — размерность векторов
        self.index = faiss.IndexIVFFlat(quantizer, dim, nlist)  # nlist — число кластеров
        self.index.train(emb_np)  # Обучение
        self.index.add(emb_np)
        self.index.nprobe = 10

        return self.index

    def search(self, queries, top_k=5):
        assert self.index is not None, "FAISS index not built. Call build_corpus first."

        q_emb = self.encode([queries]).numpy().astype('float32')
        distances, indices = self.index.search(q_emb, top_k)
        return distances, indices

    def encode(self, texts, batch_size=32):
        all_embeddings = []
        self.model.eval()
        with torch.inference_mode():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                encoded = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    return_tensors='pt'
                ).to(self.device)
                outputs = self.model(**encoded)
                hidden_states = outputs.last_hidden_state
                mask = encoded['attention_mask'].unsqueeze(-1)
                weighted = hidden_states * mask
                summed = weighted.sum(dim=1)
                counts = mask.sum(dim=1)
                embeddings = summed / counts
                all_embeddings.append(embeddings.cpu())
        return torch.cat(all_embeddings)

    def write_index_file(self, index_path):
        faiss.write_index(self.index, index_path)

    def read_index_file(self, index_path):
        self.index = faiss.read_index(index_path)
        self.index.nprobe = 10
