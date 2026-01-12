"""Minimal SymPy examples for symbolic computation

This script demonstrates basic SymPy capabilities including:
- Expression evaluation
- Symbolic differentiation and integration
- Equation solving
- Simplification and expansion
"""

from sympy import symbols, diff, integrate, solve, simplify, expand, factor
from sympy import sin, cos, exp, log, sqrt, pi, E
from sympy import pprint


def main():
    print("=" * 60)
    print("SymPy Minimal Examples")
    print("=" * 60)

    # Example 1: Expression Evaluation
    print("\n1. Expression Evaluation")
    print("-" * 40)
    x, y = symbols('x y')
    expr = x**2 + 2*x + 1
    print(f"Expression: {expr}")
    result = expr.subs(x, 3)
    print(f"Evaluated at x=3: {result}")

    # Multiple substitutions
    expr2 = x**2 + y**2
    result2 = expr2.subs([(x, 1), (y, 2)])
    print(f"Expression: {expr2}")
    print(f"Evaluated at x=1, y=2: {result2}")

    # Example 2: Symbolic Differentiation
    print("\n2. Symbolic Differentiation")
    print("-" * 40)
    expr3 = x**3 + 2*x**2 - 5*x + 3
    derivative = diff(expr3, x)
    print(f"f(x) = {expr3}")
    print(f"f'(x) = {derivative}")

    # Higher order derivatives
    second_derivative = diff(expr3, x, 2)
    print(f"f''(x) = {second_derivative}")

    # Derivative of trig functions
    expr4 = sin(x) * exp(x)
    print(f"\nd/dx[sin(x) * e^x] = {diff(expr4, x)}")

    # Example 3: Symbolic Integration
    print("\n3. Symbolic Integration")
    print("-" * 40)
    expr5 = x**2 + 2*x
    integral = integrate(expr5, x)
    print(f"∫({expr5})dx = {integral}")

    # Definite integral
    definite = integrate(x**2, (x, 0, 1))
    print(f"∫₀¹ x² dx = {definite}")

    # Example 4: Solving Equations
    print("\n4. Solving Equations")
    print("-" * 40)

    # Quadratic equation
    eq1 = x**2 - 4
    solutions1 = solve(eq1, x)
    print(f"Solve {eq1} = 0: {solutions1}")

    # System of equations
    eq2 = x + y - 5
    eq3 = x - y - 1
    solutions2 = solve([eq2, eq3], [x, y])
    print(f"Solve system: {eq2} = 0, {eq3} = 0")
    print(f"Solutions: {solutions2}")

    # Example 5: Simplification and Expansion
    print("\n5. Simplification and Expansion")
    print("-" * 40)

    # Expansion
    expr6 = (x + 1)**3
    expanded = expand(expr6)
    print(f"Expand {expr6}: {expanded}")

    # Factorization
    expr7 = x**2 - 4
    factored = factor(expr7)
    print(f"Factor {expr7}: {factored}")

    # Simplification
    expr8 = (x**2 + 2*x + 1) / (x + 1)
    simplified = simplify(expr8)
    print(f"Simplify {expr8}: {simplified}")

    # Example 6: Pretty Printing
    print("\n6. Pretty Printing (using pprint)")
    print("-" * 40)
    expr9 = integrate(x**2 * exp(-x), (x, 0, pi))
    print("Integral of x² * e^(-x) from 0 to π:")
    pprint(expr9)

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
