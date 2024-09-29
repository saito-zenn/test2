
from langchain_core.tools import tool
from langchain.document_loaders import CSVLoader
import sqlite3
import pandas as pd

conn = sqlite3.connect('cats.db')
# サンプルCSVの読み込みとSQLiteへのインポート
loader = CSVLoader(file_path="neko.csv")
data = loader.load()
df = pd.DataFrame(data)
df.to_sql('cats', conn, if_exists='replace', index=False)


# SQL実行ツール
class SQLTool:
    def __init__(self):
        self.db = sqlite3.connect('cats.db')

    @tool
    def run(self, query):
        cursor = self.db.cursor()
        cursor.execute(query)
        results = cursor.fetchall()
        return results
    