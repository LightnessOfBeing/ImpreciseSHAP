import math
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import tqdm as tqdm

from impreciseshap.solver import (
    calc_constants_dict,
    calc_constants_dict_simple,
    solve_bounds_min,
    solve_max,
    solve_min
)
from impreciseshap.utils import powerset


class ImpreciseShap:
    def __init__(
        self,
        model: Any,
        masker: pd.DataFrame,
        eps: float = 0.15,
        result_path: str = "../result.csv",
    ) -> None:
        self.model = model
        self.masker = masker
        self.means = dict(
            (col_name, self.masker[col_name].mean()) for col_name in masker.columns
        )
        self.feature_names = self.masker.columns
        self.expectation = model(self.masker).mean(axis=0)
        self.result_path = result_path
        self.eps = eps
        return

    def calculate_shapley_values(
        self, test_samples: pd.DataFrame, num_samples: int = None
    ) -> pd.DataFrame:
        test_samples = test_samples.reset_index(drop=True)
        if num_samples is not None:
            test_samples = test_samples[:num_samples]
        subsets = powerset(test_samples.columns)

        f_vals: Dict[str, List[float]] = dict((f, []) for f in self.feature_names)
        f_vals_l: Dict[str, List[float]] = dict((f, []) for f in self.feature_names)
        f_vals_u: Dict[str, List[float]] = dict((f, []) for f in self.feature_names)
        f_vals_l_c: Dict[str, List[float]] = dict((f, []) for f in self.feature_names)
        f_vals_u_c: Dict[str, List[float]] = dict((f, []) for f in self.feature_names)

        if len(test_samples) > 1:
            if num_samples is not None:
                iterator = tqdm.tqdm(test_samples.index[:num_samples])
            else:
                iterator = tqdm.tqdm(test_samples.index)
        else:
            if num_samples is not None:
                iterator = test_samples.index[:num_samples]
            else:
                iterator = test_samples.index

        new_train = self.masker.copy()
        p0 = self.expectation.cumsum()
        for index in iterator:
            shapley_values: Dict[str, float] = dict()
            shapley_values_l: Dict[str, float] = dict()
            shapley_values_u: Dict[str, float] = dict()
            for col_name in self.feature_names:
                shapley_values[col_name] = 0
                shapley_values_l[col_name] = 0
                shapley_values_u[col_name] = 0

            ts = test_samples.loc[index, :]
            model_pred = self.model(np.array(ts).reshape(1, -1))[0]
            pn = model_pred.cumsum()
            for f in self.feature_names:
                for start_index, s in enumerate(subsets):
                    if f not in s:
                        for union_index, s_union in enumerate(subsets):
                            if len(s_union) == len(s) + 1:
                                diff = list(set(s_union) - set(s))
                                if len(diff) == 1 and diff[0] == f:
                                    for col_name in self.feature_names:
                                        if col_name in s:
                                            new_train[col_name] = ts[col_name]
                                        else:
                                            new_train[col_name] = self.masker[col_name]
                                    ps = self.model(new_train)
                                    new_train[f] = ts[f]
                                    pus = self.model(new_train)

                                    pi = ps.mean(axis=0).cumsum()
                                    tau = pus.mean(axis=0).cumsum()
                                    alpha = pn

                                    ks_ps = np.abs(pi - alpha).max()
                                    ks_pus = np.abs(tau - alpha).max()

                                    diff_val = ks_ps - ks_pus

                                    mult_factor = (
                                        math.factorial(len(s))
                                        * math.factorial(
                                            len(self.feature_names) - len(s) - 1
                                        )
                                        / math.factorial(len(self.feature_names))
                                    )

                                    vs = calc_constants_dict(pi, tau, alpha, self.eps)
                                    lbc = []
                                    ubc = []
                                    for k in range(len(alpha) - 1):
                                        lbc.append(
                                            solve_min(
                                                problem_type="l1",
                                                vs=vs,
                                                k=k,
                                                pi=pi,
                                                alpha=alpha,
                                            )
                                        )
                                        lbc.append(
                                            solve_min(
                                                problem_type="l2",
                                                vs=vs,
                                                k=k,
                                                pi=pi,
                                                alpha=alpha,
                                            )
                                        )
                                        ubc.append(
                                            solve_max(
                                                problem_type="u1",
                                                vs=vs,
                                                k=k,
                                                tau=tau,
                                                alpha=alpha,
                                            )
                                        )
                                        ubc.append(
                                            solve_max(
                                                problem_type="u2",
                                                vs=vs,
                                                k=k,
                                                tau=tau,
                                                alpha=alpha,
                                            )
                                        )

                                    l = lambda x: x is not None
                                    lbc = list(filter(l, lbc))
                                    ubc = list(filter(l, ubc))

                                    lower_bound = sorted(lbc, key=lambda x: -x[1])[0][0]
                                    upper_bound = sorted(ubc, key=lambda x: -x[1])[0][0]

                                    shapley_values[f] += diff_val * mult_factor
                                    shapley_values_l[f] += lower_bound * mult_factor
                                    shapley_values_u[f] += upper_bound * mult_factor

            pi = p0
            alpha = pn
            dlc = solve_bounds_min("min", pi, alpha, self.eps)
            dl = dlc[0]

            pi = p0
            alpha = pn
            pad = calc_constants_dict_simple(pi, alpha, eps=self.eps)
            du = -math.inf
            for i in range(len(pi) - 1):
                ft = pad[f"pi_u_{i}"] - pad[f"alpha_l_{i}"]
                st = pad[f"alpha_u_{i}"] - pad[f"pi_l_{i}"]
                du = max(max(ft, st), du)

            sum_phi = 0.0
            for f_name in shapley_values:
                sum_phi += shapley_values[f_name]
                f_vals[f_name].append(shapley_values[f_name])
                f_vals_l[f_name].append(shapley_values_l[f_name])
                f_vals_u[f_name].append(shapley_values_u[f_name])

            for f_name in shapley_values:
                f_vals_l_c[f_name].append(f_vals_l[f_name][-1])
                f_vals_u_c[f_name].append(f_vals_u[f_name][-1])

            sl = 0.0
            su = 0.0
            for f_name in shapley_values:
                sl += f_vals_l[f_name][-1]
                su += f_vals_u[f_name][-1]

            for f_name in shapley_values:
                f_vals_u_c[f_name][-1] = min(
                    f_vals_u[f_name][-1], du - sl + f_vals_l[f_name][-1]
                )
                f_vals_l_c[f_name][-1] = max(
                    f_vals_l[f_name][-1], dl - su + f_vals_u[f_name][-1]
                )
                if f_vals_u_c[f_name][-1] + 1e-6 < f_vals_l_c[f_name][-1]:
                    f_vals_l_c[f_name][-1] = f_vals_l[f_name][-1]
                    f_vals_u_c[f_name][-1] = f_vals_u[f_name][-1]

        result = test_samples
        for f_name in f_vals:
            result[f"phi({f_name})_l"] = f_vals_l[f_name]
            result[f"phi({f_name})_u"] = f_vals_u[f_name]
            result[f"phi({f_name})_m"] = (
                result[f"phi({f_name})_l"]
                + (result[f"phi({f_name})_u"] - result[f"phi({f_name})_l"]) / 2
            )
            result[f"phi({f_name})"] = f_vals[f_name]
            result[f"phi({f_name})_len"] = (
                result[f"phi({f_name})_u"] - result[f"phi({f_name})_l"]
            )
            result[f"phi({f_name})_lc"] = f_vals_l_c[f_name]
            result[f"phi({f_name})_uc"] = f_vals_u_c[f_name]
            result[f"phi({f_name})_lenc"] = (
                result[f"phi({f_name})_uc"] - result[f"phi({f_name})_lc"]
            )

        result.to_csv(self.result_path, index=False)
        return result
