"""
Example of refactoring a "dirty" code-base:
- Elimination of global variables
- Use of clear names and short functions
- Removal of duplication and dead code
- Simplification of nested structures
- Separation of logic/calculation from input/output
- NumPy-style docstrings
"""

def calculate_values(list_x, list_y, increment):
    """
    Calculates a list of values based on defined rules for elements of list_x and list_y.

    Parameters
    ----------
    list_x : list of int or float
        List of first input values.
    list_y : list of int or float
        List of second input values (same length as list_x).
    increment : int or float
        Constant value to add in certain conditions.

    Returns
    -------
    list of int or float
        List of calculated values.
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

def print_analysis(values):
    """
    Analyzes and prints specific messages about the calculated values.

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

def all_positive(*args):
    """
    Checks if all provided values are positive.

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

def calculate_total(lst):
    """
    Calculates the total of a list by adding positives and subtracting negatives.

    Parameters
    ----------
    lst : list of int or float
        List of values.

    Returns
    -------
    int or float
        Calculated total.
    """
    total = 0
    for el in lst:
        if el > 0:
            total += el
        else:
            total -= el
    return total

def print_report(lst):
    """
    Prints various analyses on the elements of the list.

    Parameters
    ----------
    lst : list of int or float
        List of values.

    Returns
    -------
    None
    """
    for el in lst:
        if el % 2 == 0:
            print("Even:", el)
        else:
            print("Odd:", el)
    for el in lst:
        print("Element:", el)
    for el in lst:
        if el > 100:
            print("Very large:", el)

def main():
    # Example of running the complete program
    x = [3, 10, 7, 21]
    y = [8, 6, 9, 1]
    increment = 10

    values = calculate_values(x, y, increment)  # Main logic, short and clear function
    print_analysis(values)                      # Output separation
    print("All positive?", all_positive(*x, *y, increment))
    total = calculate_total(values)
    print("Total:", total)
    print_report(values)

if __name__ == "__main__":
    main()