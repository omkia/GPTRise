import pymysql

conn = pymysql.connect(host="localhost", user="root", password="", db="stack")

STEP = 100_000
min_id = 4
max_id = 78253176

cur = conn.cursor()

for start in range(min_id, max_id+1, STEP):
    end = start + STEP - 1
    print("Processing:", start, end)

    cur.execute("""
        INSERT INTO questions
        SELECT id, posttypeid, parentid, acceptedanswerid, creationdate, score, viewcount,
               owneruserid, lasteditoruserid, lasteditdate, lastactivitydate, tags,
               tagcount, tag_entropy, title, word_count, sentence_count, flesch_reading,
               code_block_count, code_total_lines, has_link, external_link_count,
               has_image, guessed_lang, category, confidence
        FROM posts_features
        WHERE posttypeid = 1 AND id BETWEEN %s AND %s
    """, (start, end))

    cur.execute("""
        INSERT INTO answers
        SELECT id, posttypeid, parentid, acceptedanswerid, creationdate, score, viewcount,
               owneruserid, lasteditoruserid, lasteditdate, lastactivitydate, tags,
               tagcount, tag_entropy, title, word_count, sentence_count, flesch_reading,
               code_block_count, code_total_lines, has_link, external_link_count,
               has_image, guessed_lang, category, confidence
        FROM posts_features
        WHERE posttypeid = 2 AND id BETWEEN %s AND %s
    """, (start, end))

    conn.commit()
