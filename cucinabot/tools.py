from langchain_core.tools import tool


@tool
def unit_converter_tool(value: float, from_unit: str, to_unit: str) -> str:
    """Converts a temperature, weight or volume value from one unit to another.
    Possible units are: lbs (pound), kg, g (grams), oz (ounces), pint, ml, l (liters), cup, F, C, hr, min, cm, inch.
    Args:
        value (float): The value to convert. 
        from_unit (str): The unit to convert from.
        to_unit (str): The unit to convert to.
    Returns:
        str: The converted value as a string.
    """
    print(f'-- Unit converter tool called, converting {value} from {from_unit} to {to_unit}')
    converter = {'lbs': {'kg': 0.453592, 'g': 453.592},
                 'inch': {'cm': 2.54}, 'cm': {'inch': 0.393701},
                 'pint': {'ml': 473.176, 'l': 0.473176, 'cup': 2}, 'cup': {'ml': 236.588, 'l': 0.236588, 'pint': 0.5},
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