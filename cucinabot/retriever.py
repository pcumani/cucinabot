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

bm25_retriever = BM25Retriever.from_documents(docs, k=10)

@tool
def recipes_info_tool(query: str) -> str:
    """Retrieves detailed recipes from a local Epicurious database, based on ingredients or title.
    
    Args:
        query (str): A query string containing ingredients or recipe title.
                     A '-' character can be used to not include certain ingredients, these needs to be added at the end of the query.
                     e.g., "chicken, garlic, -onion -chickpeas" to find recipes with chicken and garlic but without onion or chickpeas.
    Returns:
        str: A string containing the recipe information for at max 3 recipes."""
    print(f"-- Extracting recipe information for query: {query}")
    if '-' in query:
        ingredients = [ing.strip() for ing in query.split('-')]
        exclude_ings = ingredients[1:]
        query = ingredients[0]
    else:
        exclude_ings = []
    results = bm25_retriever.invoke(query)
    if results:
        if len(exclude_ings) > 0:
            filtered_results = []
            exluded_results = []
            for doc in results:
                if not any(ex_ing.lower() in doc.page_content.lower() for ex_ing in exclude_ings):
                    filtered_results.append(doc)
                elif len(exluded_results) < 3:
                    exluded_results.append(doc)
            
            print(f"\tFound {len(results)} matching recipes, {len(filtered_results)} after exclusion: {[doc.page_content.split('\n')[0] for doc in results[:3]]}")
            if len(filtered_results) < 3:
                recipes = "\n\n---\n\n".join([doc.page_content for doc in filtered_results]) + "\n\n---\n\n" if len(filtered_results) > 0 else ""
                recipes += "ALL THE FOLLOWING RECIPES CONTAIN AN EXCLUDED INGREDIENTS\n\n" + \
                    "\n\n---\n\n".join([doc.page_content for doc in exluded_results[:3-len(filtered_results)]])
                return recipes
            else:
                return "\n\n---\n\n".join([doc.page_content for doc in filtered_results[:3]])

        else:
            print(f"\tFound {len(results)} matching recipes: {[doc.page_content.split('\n')[0] for doc in results[:3]]}")
            return "\n\n---\n\n".join([doc.page_content for doc in results[:3]])
    else:
        print("\tNo matching recipe found.")
        return "No matching recipe found."