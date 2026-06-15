"""nugap: the Vinnicombe nu-gap metric and a pipeline for comparing
time-course data across two conditions.

Quick start
-----------
    from nugap import tf, nu_gap
    d = nu_gap(tf([1],[1,1]), tf([1],[1,1.2]))   # ~0.07

    from nugap import compare_conditions
    df = compare_conditions(data_A, data_B, t)   # ranked table of changes
"""

from .systems import LTI, tf, from_zpk, from_control, to_continuous
from .metric import nu_gap, chordal_distance, winding_condition
from .fitting import fit_model, fit_prony, fit_arx, fit_arx_fast, fit_first_order, dc_gain, FitResult
from .pipeline import compare_conditions, compare_variable
from .replicates import compare_conditions_replicates, compare_variable_replicates
from .network import compare_network

__all__ = [
    "LTI", "tf", "from_zpk", "from_control", "to_continuous",
    "nu_gap", "chordal_distance", "winding_condition",
    "fit_model", "fit_prony", "fit_arx", "fit_arx_fast", "fit_first_order", "dc_gain", "FitResult",
    "compare_conditions", "compare_variable",
    "compare_conditions_replicates", "compare_variable_replicates",
    "compare_network",
]

__version__ = "0.5.0"
