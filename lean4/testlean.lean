import Mathlib

example (x : ℝ) : x^2 + 1 ≥ 0 := by
  -- a square is always non-negative
  have h := sq_nonneg x
  -- adding 1 keeps it non-negative
  linarith [h]
