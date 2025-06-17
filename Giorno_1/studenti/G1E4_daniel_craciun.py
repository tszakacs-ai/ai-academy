"""
Module for value computation and analysis.

This module provides functions to compute values based on input lists,
analyze positivity, and print reports. All functions follow clean code
principles and include NumPy-style docstrings.
"""

def compute_values(list_x, list_y, increment):
    """
    Compute a list of values based on rules applied to elements of list_x and list_y.

    Parameters
    ----------
    list_x : list of int or float
        First input values.
    list_y : list of int or float
        Second input values (same length as list_x).
    increment : int or float
        Constant value to add in certain conditions.

    Returns
    -------
    list of int or float
        Computed values.
    """
    results = []
    for x, y in zip(list_x, list_y):
        if x > 10:
            result = x + y + increment
        elif y > 5:
            result = x * 2 + y
        else:
            result = 0
        results.append(result)
    return results

def all_positive(*args):
    """
    Check if all provided values are positive.

    Parameters
    ----------
    *args : int or float
        Any number of numeric arguments.

    Returns
    -------
    bool
        True if all values are positive, False otherwise.
    """
    return all(a > 0 for a in args)

def analyze_numbers(numbers):
    """
    Analyze a list of numbers: calculate total, print parity, print each element,
    and highlight very large numbers.

    Parameters
    ----------
    numbers : list of int or float
        List of values.

    Returns
    -------
    int or float
        Calculated total (sum of positives minus negatives).
    """
    total = 0
    for el in numbers:
        if el > 0:
            total += el
        else:
            total -= el

    for el in numbers:
        if el % 2 == 0:
            print("Even:", el)
        else:
            print("Odd:", el)

    for el in numbers:
        print("Element:", el)

    for el in numbers:
        if el > 100:
            print("Very large:", el)

    return total

def print_analysis(values):
    """
    Print analysis messages for a list of values.

    Parameters
    ----------
    values : list of int or float
        List of values to analyze.

    Returns
    -------
    None
    """
    for value in values:
        if value > 50:
            print("Large value found!")
        elif value == 0:
            print("Zero found!")
        else:
            print(f"Value: {value}")

def main():
    """
    Main execution function.
    """
    x = [5, 12, 8, 15]
    y = [6, 4, 7, 2]
    increment = 10

    values = compute_values(x, y, increment)
    print_analysis(values)
    print("All positive?", all_positive(*x, *y, increment))
    total = analyze_numbers(values)
    print("Total:", total)

if __name__ == "__main__":
    main()