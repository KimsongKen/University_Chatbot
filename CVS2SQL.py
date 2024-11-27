import sqlite3
import pandas as pd

df = pd.read_csv('vme_chatbot - Majors.csv')

df.columns = df.columns.str.strip()

connection = sqlite3.connect('vme_chatbot - Majors.db')

df.to_sql('vme_chatbot - Majors.db', connection, if_exists='replace')

#close the connection
connection.close()
