import re
from util import connect_db
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
conn, cursor = connect_db("localhost", "root", "", "deteksi_trending_topik")

factory = StemmerFactory()
stemmer = factory.create_stemmer()

stop_factory = StopWordRemoverFactory()
stopword = stop_factory.create_stop_word_remover()

cursor.execute("SELECT * FROM dokumen")
raw_text = cursor.fetchall()

cursor.execute("Select * FROM kamus")
kamus = cursor.fetchall()

for x in raw_text:
    bersih = re.sub('[^a-zA-Z]+', ' ', x[1])
    
    bersih = re.sub(' +', ' ', bersih)
    bersih = bersih.strip()

    bersih = bersih.lower()
    
    s = ''
    bersih = bersih.split()

    for y in bersih:
        for x1 in kamus:
            if y == x1[1] :
                y = x1[2]
        s = s + y + " "
        bersih = s 
    
    bersih = stemmer.stem(str(bersih))
    bersih = stopword.remove(bersih)

    cursor.execute("UPDATE dokumen set preproccess_text=%s WHERE id = %s", (bersih, x[0]))

conn.commit()

    
