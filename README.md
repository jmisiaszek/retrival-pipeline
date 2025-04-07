# retrival-pipeline
A task for JetBrains Internship recrutation

A pipeline for chunk information retrival and quality metrics.

## Files

Quick description of souce files in this project

- `trimCSV.py` Used to trim file with questions to only those related to present corporas.
- `FixedTokenChunker.py` Implementation of the chunker.
- `chunk_eval.py` Implementation of the pipeline.

## How to run

Install all necessary dependencies using 

```
pip install -r requirements.txt
```

and then run the script using 

```
python3 chunk_eval.py > result
```

## Results and summary

Here's the table with results that I got:

| Parameters                  | Mean Precision          | Std Precision         | Mean Recall            | Std Recall             |
|-----------------------------|-------------------------|-----------------------|------------------------|------------------------|
| chunk_size=400, chunk_num=10 | 0.0018051932349533778    | 0.005289221104898678   | 0.1202822079877414      | 0.3274033320325772      |
| chunk_size=400, chunk_num=5  | 0.009132474129943615     | 0.018678100316375578   | 0.28664758283912994     | 0.4737632880451287      |
| chunk_size=200, chunk_num=10 | 0.011973157208216388     | 0.02122492200450722    | **0.4030443037628191**      | 0.5633036074303283      |
| chunk_size=200, chunk_num=5  | **0.021467239166554575**     | 0.034981500455698804   | 0.35135880395593744     | 0.4646206938649207      |

Intuitively, reducing the chunk size resulted in higher mean Precision value. This comes from less text being selected overall. What also plays in to this score is golden excerpts being quite short on average.

The unexpected higher Recall in second row in comparison to the first might come from embedding model. I looked further in the results and indeed some chunks in second run had higher intersection than in the first.

There's a tradeoff between Precision and Recall, which comes from including more or less text in the context. This is only visible in row 4, where the Recall value started to fall down. This might mean that we crossed the line, and further lowering of both parameters might worsen the Recall even more.