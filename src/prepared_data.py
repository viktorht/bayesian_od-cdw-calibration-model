""""Class defining what prepared data looks like."""


from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import pandas as pd

from src.util import CoordDict, StanInput


@dataclass
class PreparedData:
    name: str
    coords: CoordDict
    dims: Dict[str, Any]
    number_of_cv_folds: int
    stan_input_function: Callable[..., StanInput]
    measurements: Optional[pd.DataFrame] = None # used for mean_model
    measurements_od: Optional[pd.DataFrame] = None
    measurements_cdw: Optional[pd.DataFrame] = None
