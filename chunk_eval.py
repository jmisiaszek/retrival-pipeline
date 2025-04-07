import csv
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import (
    Any,
    List,
    Dict,
    Callable
)
from FixedTokenChunker import TextSplitter, FixedTokenChunker
from sentence_transformers import SentenceTransformer

def find_edges(chunks: List[str], corpus: str):
    """ Find begin and end index of chunk in the whole document. """
    edges = []
    for chunk in chunks:
        start = corpus.find(chunk)
        if start == -1:
            raise Exception("Chunk not in the corpus.")
        end = start + len(chunk)
        edges.append({"start": start, "end": end})
    return edges

def make_collection(chunks: List[str], embeds: List[float], corpus: str):
    """ Compile chunks and metadata into one collection. """
    edges = find_edges(chunks, corpus)
    metadata = []
    for chunk, embed, edge in zip(chunks, embeds, edges):
        metadata.append({
            "chunk": chunk,
            "embed": embed,
            "start": edge["start"],
            "end":   edge["end"]
        })
    return metadata

def get_num_chunks(question: str, 
                   collection: List[Dict[str, Any]], 
                   embed_fn: Callable[[List[str]], List[float]], 
                   num: int):
    """ Retrive num first chunks using cosine similarity. """

    q_embed = embed_fn([question])
    embeds = [chunk["embed"] for chunk in collection]

    similarity = cosine_similarity(q_embed, embeds)

    indices = np.argsort(similarity).reshape(-1)
    top_num = indices[-num:][::-1]

    top_chunks = [collection[idx] for idx in top_num]
    return top_chunks

def evaluate(chunker: TextSplitter, 
             embed_fn: Callable[[List[str]], List[float]], 
             chunk_num: int):
    """
    Run the chunking evaluation pipeline.
    
    Args:
        chunker: Chunker of TextSplitter type used to split text.
        embed_fn: Embedding function.
        chunk_num: Number of top chunks to retrive.
    """
    # I picked "wikitexts.md" as my corpus because it has the biggest size.
    corpus_file = open("./corpora/wikitexts.md", 'r')
    corpus = corpus_file.read()
    questions = open("./questions_df_trimmed.csv", 'r')
    
    chunks_text = chunker.split_text(corpus)
    embeds = embed_fn(chunks_text)

    collection = make_collection(chunks_text, embeds, corpus)

    q_reader = csv.DictReader(questions)

    for row in q_reader:
        if row["corpus_id"] == "wikitexts":
            top_chunk = get_num_chunks(row["question"], collection, embed_fn, chunk_num)



            print(row["question"], "\n")
            for chunk in top_chunk:
                print(chunk["chunk"], "\n")
            print("----\n")
            


if __name__ == "__main__":
    chunker = FixedTokenChunker(chunk_size=400, chunk_overlap=20)

    def calculate_embeds(chunks: List[str]) -> List[float]:
        """ Calculate embeddings for chunks. """
        embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        embeds = embed_model.encode(chunks)
        return embeds
        
    evaluate(chunker, calculate_embeds, 10)