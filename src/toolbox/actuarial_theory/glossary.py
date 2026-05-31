"""Actuarial terminology dictionary.

Created: 2026-05-31
Purpose: Store core actuarial terms referenced by the platform and documentation.
"""

from __future__ import annotations


GLOSSARY: dict[str, str] = {
    "thiele_equation": "A differential equation describing reserve evolution under interest, premiums, and decrements.",
    "reserve": "Present value of future liabilities minus future premiums under chosen assumptions.",
    "sum_assured": "Policy benefit paid on insured death during the term of coverage.",
}
