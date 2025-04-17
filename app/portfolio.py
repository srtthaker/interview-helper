import pandas as pd
import chromadb
import uuid


class Portfolio:
    def __init__(self, file_path="app/resource/my_portfolio.csv"):
        self.file_path = file_path
        self.data = pd.read_csv(file_path)
        self.chroma_client = chromadb.PersistentClient("vector_store")
        self.collection = self.chroma_client.get_or_create_collection(name="portfolio")

    def load_portfolio(self):
        if not self.collection.count():
            for _, row in self.data.iterrows():
                self.collection.add(
                    documents=row["Stack"],
                    metadatas={"description": row["Description"]},
                    ids=[str(uuid.uuid4())],
                )

    def query_portfolio(self, skills):
        return self.collection.query(query_texts=skills, n_results=2).get(
            "metadatas", []
        )
