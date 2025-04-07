import csv
from typing import (
    Any,
    List, 
    Callable
)
from FixedTokenChunker import TextSplitter, FixedTokenChunker
from sentence_transformers import SentenceTransformer

def eval(chunker: TextSplitter, embed_fn: Callable[[List[str]], List[float]], chunk_num: int):
    """Run the chunking evaluation pipeline.
    
    Args:
        chunker: Chunker of TextSplitter type used to split text.
        embed_fn: Embedding function.
        chunk_num: Number of top chunks to retrive.
    """
    # I picked "wikitexts.md" as my corpus because it has the biggest size.
    corpus = open("./corpora/wikitexts.md", 'r')
    questions = open("./questions_df_trimmed.csv", 'r')
    
    chunks : List[str] = chunker.split_text(corpus.read())
    embeds = embed_fn(chunks)

    for chunk, embed in zip(chunks, embeds):
        print(chunk)
        print("\n")
        print(embed)
        print("\n---\n")


def main():
    chunker = FixedTokenChunker()

    def calculate_embeds(chunks: List[str]):
        """ Calculate embeddings for chunks. """
        embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        embeds = embed_model.encode(chunks)
        return embeds
        
    eval(chunker, calculate_embeds, 10)
    


if __name__ == "__main__":
    main()