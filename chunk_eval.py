import csv
import numpy as np
import json
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
            raise Exception(f"Chunk not in the corpus: {chunk}")
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

def eval_question(reference_coll: List[Dict[str, Any]], 
                  chunk_coll: List[Dict[str, Any]],
                  corpus: str):
    
    size_retrived = 0
    for chunk in chunk_coll:
        size_retrived += len(chunk["chunk"])
    
    size_references = 0
    for chunk in reference_coll:
        size_references += len(chunk["content"])

    chunk_sorted = sorted(chunk_coll, key=lambda x: x["start"])
    for i in range(len(chunk_coll) - 1):
        chunk_sorted[i]["end"] = min(chunk_coll[i]["end"], chunk_coll[i + 1]["start"])

    size_intersection = 0
    for chunk in chunk_sorted:
        for reference in reference_coll:
            start = max(chunk["start"], reference["start_index"])
            end = min(chunk["end"], reference["end_index"])
            if start < end:
                size_intersection += end - start

    precision = size_intersection / size_retrived
    recall = size_intersection / size_references

    return precision, recall


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
    corpus_file = open("./corpora/wikitexts.md", 'r', encoding="utf-8")
    corpus = corpus_file.read()
    questions = open("./questions_df_trimmed.csv", 'r')
    
    chunks_text = chunker.split_text(corpus)
    embeds = embed_fn(chunks_text)

    collection = make_collection(chunks_text, embeds, corpus)

    q_reader = csv.DictReader(questions)

    precisions = []
    recalls = []

    for row in q_reader:
        if row["corpus_id"] == "wikitexts":
            top_chunk = get_num_chunks(row["question"], collection, embed_fn, chunk_num)

            precision, recall = eval_question(json.loads(row["references"]), top_chunk, corpus)
            precisions.append(precision)
            recalls.append(recall)

    print(f"Mean precision: {np.mean(precisions)}")
    print(f"Std precision: {np.std(precisions)}")
    print(f"Mean recall: {np.mean(recalls)}")
    print(f"Std recalls: {np.std(recalls)}")

if __name__ == "__main__":
    # Chunk overlap will be around 5% of the chunk size
    chunker_400 = FixedTokenChunker(chunk_size=400, chunk_overlap=20)
    chunker_200 = FixedTokenChunker(chunk_size=200, chunk_overlap=10)
    chunker_100 = FixedTokenChunker(chunk_size=100, chunk_overlap=6)

    def calculate_embeds(chunks: List[str]) -> List[float]:
        """ Calculate embeddings for chunks. """
        embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        embeds = embed_model.encode(chunks)
        return embeds
    
    print("Evaluation of FixedTokenChunker")

    print("\nParams: chunk_size=400, chunk_num=10\n")
    evaluate(chunker_400, calculate_embeds, 10)

    print("\nParams: chunk_size=400, chunk_num=5\n")
    evaluate(chunker_400, calculate_embeds, 5)

    print("\n")

    print("\nParams: chunk_size=200, chunk_num=10\n")
    evaluate(chunker_200, calculate_embeds, 10)

    print("\nParams: chunk_size=200, chunk_num=5\n")
    evaluate(chunker_200, calculate_embeds, 5)