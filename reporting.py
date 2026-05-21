from __future__ import annotations


def print_rule(title: str, char: str = "=", width: int = 70) -> None:
    print(char * width, flush=True)
    print(title, flush=True)
    print(char * width, flush=True)


def print_section(title: str, width: int = 70) -> None:
    print(title, flush=True)
    print("-" * width, flush=True)


def p_value_interpretation(
    p_value: float,
    alpha: float = 0.05,
    adjusted_p: float | None = None,
) -> str:
    if adjusted_p is not None:
        if p_value < alpha <= adjusted_p:
            return "marginal before correction; not significant after correction"
        if adjusted_p < alpha:
            return "significant after correction"
        return "not significant after correction"

    if p_value < alpha:
        return "significant at the stated alpha"
    return "not significant at the stated alpha"


def print_p_value(
    label: str,
    p_value: float,
    alpha: float = 0.05,
    adjusted_p: float | None = None,
    correction_label: str | None = None,
) -> None:
    print(f"{label}: p = {p_value:.4f}")
    if adjusted_p is not None:
        correction = f" ({correction_label})" if correction_label else ""
        print(f"{label} adjusted{correction}: p = {adjusted_p:.4f}")
    print(f"  Interpretation: {p_value_interpretation(p_value, alpha, adjusted_p)}")
