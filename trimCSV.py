import csv

original = open("questions_df.csv", 'r')
target = open("questions_df_trimmed.csv", 'w')

reader =  csv.DictReader(original)
writer = csv.DictWriter(target, reader.fieldnames)
writer.writeheader()

for row in reader:
    if row["corpus_id"] == "state_of_the_union" or \
       row["corpus_id"] == "wikitexts" or \
       row["corpus_id"] == "chatlogs":
        writer.writerow(row)