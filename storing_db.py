import csv
from util import connect_db

conn, cursor = connect_db("localhost", "root", "", "deteksi_trending_topik")

# Read the CSV file
with open('indonesian-news-title-200.csv', 'r', encoding='utf-8') as file:
    reader = csv.reader(file)
    header = next(reader)
    # Get the index of the second column
    judul_index = header.index('title')
    for row in reader:
        # Get the text from the second column
        judul = row[judul_index]
        # Insert the text into the database
        cursor.execute('''
            INSERT INTO dokumen (raw_text)
            VALUES (%s)
        ''', (judul,))

# Commit the changes and close the connection
conn.commit()
conn.close()
