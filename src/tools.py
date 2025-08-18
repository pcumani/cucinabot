import requests
import re
from bs4 import BeautifulSoup
from langchain_core.tools import tool
from langchain_community.document_loaders import WebBaseLoader


@tool
def get_recipes_web_tool(ingredients: str) -> str:
    """
    A list of food recipes from the Giallo Zafferano website.

    Args:
        ingredients: A comma-separated string of ingredients to search for recipes.
    Returns:
        A string containing the recipes found, formatted as XML-like documents.
    """
    print(f"-- Fetching recipes for ingredients: {ingredients}")
    
    url = f"https://www.giallozafferano.com/recipes-search/+{'+'.join([x.strip() for x in ingredients.split(',')])}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
    except Exception as e:
        print(f"Error fetching recipes: {e}")
        return f"Error fetching recipes: {str(e)}"
    
    soup = BeautifulSoup(response.content, features="lxml")
    content = soup.find_all('article', class_='gz-card')

    if len(content) == 0:
        return "No recipes found for the given ingredients."
    
    recipes = []

    if len(recipes) == 0:
        return "No recipes found for the given ingredients."

    for item in content:
        
        title = item.find('h2', class_='gz-title').get_text(strip=True)
        info = item.find('ul', class_='gz-card-data').get_text()
        info = [x.strip() for x in re.split('\n+', info.strip()) if x.strip()]
        info = f'Difficulty: {info[0] if len(info)>0 else ''}, Time: {info[1] if len(info)>1 else ''}, Servings: {info[2] if len(info)>2 else ''}'
        
        link = item.find('a')['href']

        description = item.find('div', class_='gz-description').get_text(strip=True)

        recipes.append(f'<Document source="{link}" title="{title}"/>\n{info}\n{description}\n</Document>')

    print(f'Found {len(recipes)} recipes')
    return "\n\n---\n\n".join(recipes)

@tool
def unit_converter_tool(value: float, from_unit: str, to_unit: str) -> str:
    """Converts a temperature, weight or volume value from one unit to another.
    Args:
        value (float): The value to convert. 
        from_unit (str): The unit to convert from. Possible units are: lbs, kg, g, oz, ml, l, cup, F, C, hr, min.
        to_unit (str): The unit to convert to. Possible units are: lbs, kg, g, oz, ml, l, cup, F, C, hr, min.
    Returns:
        str: The converted value as a string.
    """
    print(f'-- Unit converter tool called, converting {value} from {from_unit} to {to_unit}')
    converter = {'lbs': {'kg': 0.453592, 'g': 453.592},
                 'kg': {'lbs': 2.20462, 'g': 1000},
                 'g': {'lbs': 0.00220462, 'kg': 0.001},
                 'oz': {'g': 28.3495, 'kg': 0.0283495, 'lbs': 0.0625},
                 'ml': {'l': 0.001, 'cup': 0.00422675}, 'l': {'ml': 1000, 'cup': 4.22675}, 'cup': {'ml': 236.588, 'l': 0.236588},
                 'F': {'C': lambda f: (f - 32) * 5.0/9.0}, 'C': {'F': lambda c: c * 9.0/5.0 + 32}, 
                 'hr': {'min': 60}, 'min': {'hr': 1/60}}
    if from_unit not in converter or to_unit not in converter[from_unit]:
        return f"Conversion from {from_unit} to {to_unit} is not supported."

    if isinstance(converter[from_unit][to_unit], float):
        converted_value = value * converter[from_unit][to_unit]
    else:
        converted_value = converter[from_unit][to_unit](value)
    return f"Converted value: {converted_value:.2f} {to_unit}"

@tool
def multiply(a: float, b: float) -> float:
    """
    Multiplies two numbers.
    Args:
        a (float): the first number
        b (float): the second number
    """
    return a * b


@tool
def add(a: float, b: float) -> float:
    """
    Adds two numbers.
    Args:
        a (float): the first number
        b (float): the second number
    """
    return a + b


@tool
def subtract(a: float, b: float) -> int:
    """
    Subtracts two numbers.
    Args:
        a (float): the first number
        b (float): the second number
    """
    return a - b


@tool
def divide(a: float, b: float) -> float:
    """
    Divides two numbers.
    Args:
        a (float): the first float number
        b (float): the second float number
    """
    if b == 0:
        raise ValueError("Cannot divided by zero.")
    return a / b

@tool
def webpage_reader_tool(page_url: str) -> str:
    """A tool to read the full content of a recipe from the Giallo Zafferano website.

    Args:
        page_url (str): A valid URL of the webpage to read.
    Returns:
        str: The text content of the webpage.
    """
    print('-- Web page reader tool called', page_url)
    loader = WebBaseLoader(web_paths=[page_url])
    docs = []
    for doc in loader.lazy_load():
        docs.append(doc)

    assert len(docs) == 1
    doc = docs[0]

    return f'<Document source="{page_url}" title="{doc.get("title", "")}"/>\n{doc.page_content.strip()}\n</Document>'