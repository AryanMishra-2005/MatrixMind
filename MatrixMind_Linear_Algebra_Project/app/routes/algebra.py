"""Numerical linear-algebra workbench endpoints powered by NumPy."""
from __future__ import annotations

from typing import Literal
import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter()


class CalculateRequest(BaseModel):
    operation: Literal[
        "add", "subtract", "multiply", "transpose", "determinant", "inverse",
        "rank", "trace", "eigenvalues", "rref", "solve", "norm"
    ]
    matrix_a: list[list[float]] = Field(min_length=1, max_length=12)
    matrix_b: list[list[float]] | None = None
    vector_b: list[float] | None = None


def _matrix(values: list[list[float]], name: str) -> np.ndarray:
    try:
        arr = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric values.") from exc
    if arr.ndim != 2 or not arr.size:
        raise ValueError(f"{name} must be a non-empty 2D matrix.")
    if arr.shape[0] > 12 or arr.shape[1] > 12:
        raise ValueError("Matrices are limited to 12 rows and 12 columns in this workbench.")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains a non-finite value.")
    if len({len(row) for row in values}) != 1:
        raise ValueError(f"{name} must be rectangular (all rows need the same number of values).")
    return arr


def _clean(value):
    if isinstance(value, np.ndarray):
        return [_clean(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, complex):
        if abs(value.imag) < 1e-10:
            return round(float(value.real), 8)
        return f"{value.real:.8g}{value.imag:+.8g}i"
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return 0 if abs(number) < 1e-10 else round(number, 8)
    return value


def _rref(matrix: np.ndarray, tol: float = 1e-10):
    a = matrix.astype(float).copy()
    rows, cols = a.shape
    pivot_cols = []
    pivot_row = 0
    for col in range(cols):
        if pivot_row >= rows:
            break
        best = pivot_row + int(np.argmax(np.abs(a[pivot_row:, col])))
        if abs(a[best, col]) <= tol:
            continue
        a[[pivot_row, best]] = a[[best, pivot_row]]
        a[pivot_row] /= a[pivot_row, col]
        for row in range(rows):
            if row != pivot_row:
                a[row] -= a[row, col] * a[pivot_row]
        a[np.abs(a) < tol] = 0.0
        pivot_cols.append(col)
        pivot_row += 1
    return a, pivot_cols


@router.post("/algebra/calculate")
def calculate(req: CalculateRequest):
    try:
        a = _matrix(req.matrix_a, "Matrix A")
        b = _matrix(req.matrix_b, "Matrix B") if req.matrix_b is not None else None
        op = req.operation
        steps: list[str] = []
        title = op.replace("_", " ").title()

        if op in {"add", "subtract", "multiply"}:
            if b is None:
                raise ValueError("This operation requires Matrix B.")
            if op in {"add", "subtract"} and a.shape != b.shape:
                raise ValueError(f"Matrix A and B must have the same dimensions. Got {a.shape} and {b.shape}.")
            if op == "multiply" and a.shape[1] != b.shape[0]:
                raise ValueError(f"Columns of A ({a.shape[1]}) must equal rows of B ({b.shape[0]}).")
            if op == "add":
                result = a + b
                steps = [f"A and B are both {a.shape[0]} × {a.shape[1]}.", "Add corresponding entries: (A + B)ᵢⱼ = aᵢⱼ + bᵢⱼ."]
            elif op == "subtract":
                result = a - b
                steps = [f"A and B are both {a.shape[0]} × {a.shape[1]}.", "Subtract corresponding entries: (A − B)ᵢⱼ = aᵢⱼ − bᵢⱼ."]
            else:
                result = a @ b
                steps = [f"A is {a.shape[0]} × {a.shape[1]} and B is {b.shape[0]} × {b.shape[1]}.", "Multiply each row of A by each column of B and add the products."]
        elif op == "transpose":
            result = a.T
            steps = [f"Swap rows and columns: {a.shape[0]} × {a.shape[1]} becomes {a.shape[1]} × {a.shape[0]}.", "The entry in row i, column j moves to row j, column i."]
        elif op == "determinant":
            if a.shape[0] != a.shape[1]:
                raise ValueError("Determinant is defined only for a square matrix.")
            result = float(np.linalg.det(a))
            steps = [f"A is a square {a.shape[0]} × {a.shape[1]} matrix.", "The determinant is a scalar; a value of 0 means the matrix is singular."]
        elif op == "inverse":
            if a.shape[0] != a.shape[1]:
                raise ValueError("Inverse is defined only for a square matrix.")
            det = float(np.linalg.det(a))
            if abs(det) < 1e-10:
                raise ValueError("Matrix A is singular (determinant is zero), so it has no inverse.")
            result = np.linalg.inv(a)
            steps = [f"det(A) = {_clean(det)} ≠ 0, so A is invertible.", "The inverse satisfies A × A⁻¹ = I."]
        elif op == "rank":
            result = int(np.linalg.matrix_rank(a))
            steps = ["Reduce the matrix to row-echelon form.", "The rank is the number of pivot positions (equivalently, non-zero rows in echelon form)."]
        elif op == "trace":
            if a.shape[0] != a.shape[1]:
                raise ValueError("Trace is defined only for a square matrix.")
            result = float(np.trace(a))
            steps = ["Add the main diagonal entries of A."]
        elif op == "eigenvalues":
            if a.shape[0] != a.shape[1]:
                raise ValueError("Eigenvalues are defined only for a square matrix.")
            result = [_clean(x) for x in np.linalg.eigvals(a)]
            steps = ["Solve the characteristic equation det(A − λI) = 0.", "The roots λ are the eigenvalues of A."]
        elif op == "rref":
            result, pivots = _rref(a)
            steps = ["Use elementary row operations to create leading 1s.", "Make every pivot the only non-zero entry in its column.", f"Pivot columns (1-based): {[p + 1 for p in pivots] or 'none'}."]
        elif op == "solve":
            if req.vector_b is None:
                raise ValueError("Enter vector b to solve A x = b.")
            if a.shape[0] != a.shape[1]:
                raise ValueError("This solver currently expects a square coefficient matrix A.")
            rhs = np.asarray(req.vector_b, dtype=float)
            if rhs.ndim != 1 or rhs.shape[0] != a.shape[0]:
                raise ValueError(f"Vector b must contain {a.shape[0]} values.")
            if not np.isfinite(rhs).all():
                raise ValueError("Vector b contains a non-finite value.")
            result = np.linalg.solve(a, rhs)
            steps = ["Write the system as A x = b.", "Solve using a numerically stable linear-system solver.", "Substitute x back into A x = b to verify the solution."]
        elif op == "norm":
            result = float(np.linalg.norm(a))
            steps = ["The Frobenius norm is the square root of the sum of the squared matrix entries."]
        else:
            raise ValueError("Unsupported operation.")

        shape = list(np.asarray(result).shape) if np.asarray(result).ndim else []
        return {
            "operation": op,
            "title": title,
            "result": _clean(result),
            "result_shape": shape,
            "steps": steps,
            "matrix_a_shape": list(a.shape),
            "matrix_b_shape": list(b.shape) if b is not None else None,
        }
    except np.linalg.LinAlgError as exc:
        raise HTTPException(status_code=400, detail=f"Linear algebra calculation failed: {exc}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
