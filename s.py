import pymysql
conn = pymysql.connect(host='localhost',user='root',password='123456',database='paper')
cursor = conn.cursor()
sql = 'select title,year from paper_info where deeplearning = 0 order by year;'
cursor.execute(sql)
data = cursor.fetchall()
f = open('troditional.txt','w',encoding='utf8')
for p in data:
    f.write(p[0]+'\t'+p[1])
    f.write('\n')
f.close()