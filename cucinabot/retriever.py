import os
from pathlib import Path
from langchain_core.tools import tool
from langchain_community.retrievers import BM25Retriever
from langchain.docstore.document import Document
import datasets

# Load the dataset
data_path = str(os.path.join(Path(__file__).parent.parent,  "data", "Recipe_Dataset_Cleaned.csv"))
recipe_dataset = datasets.load_dataset("csv", data_files=data_path, column_names= ['Title', 'Instructions', 'Cleaned_Ingredients'])['train']

# Convert dataset entries into Document objects
docs = [
    Document(
        page_content="\n".join([
            f"Title: {recipe['Title']}",
            f"Ingredients: {recipe['Cleaned_Ingredients']}",
            f"Instructions: {recipe['Instructions']}"
        ]),
        metadata={"Ingredients": recipe["Cleaned_Ingredients"], "Title": recipe["Title"]},
    )
    for recipe in recipe_dataset
]

bm25_retriever = BM25Retriever.from_documents(docs)

@tool
def recipes_info_tool(query: str) -> str:
    """Retrieves detailed recipes from a local Epicurious database, based on ingredients or title.
    
    Args:
        query (str): A query string containing ingredients or recipe title.
    Returns:
        str: A string containing the recipe information for at max 3 recipes."""
    print(f"-- Extracting recipe information for query: {query}")
    results = bm25_retriever.invoke(query)
    if results:
        print(f"\tFound {len(results)} matching recipes.")
        return "\n\n---\n\n".join([doc.page_content for doc in results[:3]])
    else:
        print("\tNo matching guest information found.")
        return "No matching guest information found."