from typing import Dict, List, Optional, Tuple

import numpy as np
from pulp import LpMaximize, LpMinimize, LpProblem, LpVariable


def calc_constants_dict(
    pi: np.ndarray, tau: np.ndarray, alpha: np.ndarray, eps: float
) -> Dict[str, float]:
    vals = dict()
    for index in range(len(pi) - 1):
        vals[f"alpha_l_{index}"] = (1 - eps) * alpha[index]
        vals[f"alpha_u_{index}"] = (1 - eps) * alpha[index] + eps
        vals[f"pi_l_{index}"] = (1 - eps) * pi[index]
        vals[f"pi_u_{index}"] = (1 - eps) * pi[index] + eps
        vals[f"tau_l_{index}"] = (1 - eps) * tau[index]
        vals[f"tau_u_{index}"] = (1 - eps) * tau[index] + eps
    return vals


def calc_constants_dict_simple(
    pi: np.ndarray, alpha: np.ndarray, eps: float
) -> Dict[str, float]:
    vals = dict()
    for index in range(len(pi) - 1):
        vals[f"pi_l_{index}"] = (1 - eps) * pi[index]
        vals[f"pi_u_{index}"] = (1 - eps) * pi[index] + eps
        vals[f"alpha_l_{index}"] = (1 - eps) * alpha[index]
        vals[f"alpha_u_{index}"] = (1 - eps) * alpha[index] + eps
    return vals


def solve_min_l1(
    vs: Dict[str, float], k: int, pi: np.ndarray, alpha: np.ndarray
) -> Optional[Tuple[float, float, int]]:
    model = LpProblem(name=f"{k}", sense=LpMinimize)
    B = LpVariable(name="B")
    pis = [LpVariable(f"pi_{i}") for i in range(len(pi) - 1)]
    alphas = [LpVariable(f"alpha_{i}") for i in range(len(alpha) - 1)]
    model += B - vs[f"tau_u_{k}"] + alphas[k]
    C = len(alpha)

    for i in range(C - 1):
        model += pis[i] >= vs[f"pi_l_{i}"]
        model += pis[i] <= vs[f"pi_u_{i}"]
        model += alphas[i] >= vs[f"alpha_l_{i}"]
        model += alphas[i] <= vs[f"alpha_u_{i}"]

    for i in range(C - 1):
        model += B >= pis[i] - alphas[i]
        model += B >= alphas[i] - pis[i]

    for i in range(C - 2):
        model += pis[i] <= pis[i + 1]
        model += alphas[i] <= alphas[i + 1]
    for i in range(C - 1):
        if i != k:
            model += vs[f"tau_u_{k}"] - alphas[k] >= vs[f"tau_u_{i}"] - alphas[i]

    model += alphas[k] <= vs[f"tau_u_{k}"]

    status = model.solve()
    if status == 1:
        diff = vs[f"tau_u_{k}"] - alphas[k].value()
        return model.objective.value(), diff, k
    return None


def solve_min_l2(
    vs: Dict[str, float], k: int, pi: np.ndarray, alpha: np.ndarray
) -> Optional[Tuple[float, float, int]]:
    model = LpProblem(name=f"{k}", sense=LpMinimize)
    B = LpVariable(name="B")
    pis = [LpVariable(f"pi_{i}") for i in range(len(pi) - 1)]
    alphas = [LpVariable(f"alpha_{i}") for i in range(len(alpha) - 1)]
    model += B - alphas[k] + vs[f"tau_l_{k}"]
    C = len(alpha)

    for i in range(C - 1):
        model += pis[i] >= vs[f"pi_l_{i}"]
        model += pis[i] <= vs[f"pi_u_{i}"]
        model += alphas[i] >= vs[f"alpha_l_{i}"]
        model += alphas[i] <= vs[f"alpha_u_{i}"]

    for i in range(C - 1):
        model += B >= pis[i] - alphas[i]
        model += B >= alphas[i] - pis[i]

    for i in range(C - 2):
        model += pis[i] <= pis[i + 1]
        model += alphas[i] <= alphas[i + 1]

    for i in range(C - 1):
        if i != k:
            model += alphas[k] - vs[f"tau_l_{k}"] >= alphas[i] - vs[f"tau_l_{i}"]

    model += alphas[k] >= vs[f"tau_l_{k}"]

    status = model.solve()
    if status == 1:
        diff = alphas[k].value() - vs[f"tau_l_{k}"]
        return model.objective.value(), diff, k
    return None


def solve_max(
    type: str, vs: Dict[str, float], k: int, tau: np.ndarray, alpha: np.ndarray
) -> Optional[Tuple[float, float, int]]:
    model = LpProblem(name=f"{type}-{k}", sense=LpMaximize)
    B = LpVariable(name="B")
    taus = [LpVariable(f"tau_{i}") for i in range(len(tau) - 1)]
    alphas = [LpVariable(f"alpha_{i}") for i in range(len(alpha) - 1)]
    if type == "u1":
        model += vs[f"pi_u_{k}"] - alphas[k] - B
    else:
        model += alphas[k] - vs[f"pi_l_{k}"] - B
    C = len(alpha)

    for i in range(C - 1):
        model += taus[i] >= vs[f"tau_l_{i}"]
        model += taus[i] <= vs[f"tau_u_{i}"]
        model += alphas[i] >= vs[f"alpha_l_{i}"]
        model += alphas[i] <= vs[f"alpha_u_{i}"]

    for i in range(C - 1):
        model += B >= taus[i] - alphas[i]
        model += B >= alphas[i] - taus[i]

    for i in range(C - 2):
        model += taus[i] <= taus[i + 1]
        model += alphas[i] <= alphas[i + 1]

    if type == "u1":
        model += alphas[k] <= vs[f"pi_u_{k}"]
    else:
        model += alphas[k] >= vs[f"pi_l_{k}"]

    status = model.solve()
    if status == 1:
        if type == "u1":
            diff = vs[f"pi_u_{k}"] - alphas[k].value()
        else:
            diff = alphas[k].value() - vs[f"pi_l_{k}"]
        return model.objective.value(), diff, k
    return None


def solve_bounds_min(
    type: str, pi: np.ndarray, alpha: np.ndarray, eps: float
) -> List[np.float64]:
    answers = []
    constants = calc_constants_dict_simple(pi, alpha, eps)
    C = len(pi)

    pis = [LpVariable(f"pi_{i}") for i in range(C - 1)]
    alphas = [LpVariable(f"alpha_{i}") for i in range(C - 1)]
    B = LpVariable(name="B")
    model = LpProblem(name=f"{type}", sense=LpMinimize)
    model += B

    for i in range(C - 1):
        model += B >= pis[i] - alphas[i]
        model += B >= alphas[i] - pis[i]
        model += pis[i] >= constants[f"pi_l_{i}"]
        model += pis[i] <= constants[f"pi_u_{i}"]
        model += alphas[i] >= constants[f"alpha_l_{i}"]
        model += alphas[i] <= constants[f"alpha_u_{i}"]

    for i in range(C - 2):
        model += pis[i] <= pis[i + 1]
        model += alphas[i] <= alphas[i + 1]

    status = model.solve()
    if status == 1:
        answers.append(model.objective.value())

    return sorted(answers)


def solve_bounds_max(
    type: str, pi: np.ndarray, alpha: np.ndarray, eps: float
) -> List[np.float64]:
    answers = []
    constants = calc_constants_dict_simple(pi, alpha, eps)
    C = len(pi)
    for k in range(C - 1):
        pis = [LpVariable(f"t_{i}") for i in range(C - 1)]
        alphas = [LpVariable(f"q_{i}") for i in range(C - 1)]
        model = LpProblem(name=f"{type}-{k}", sense=LpMaximize)

        model += pis[k] - alphas[k]
        model += pis[k] >= alphas[k]

        for i in range(C - 1):
            model += pis[i] >= constants[f"pi_l_{i}"]
            model += pis[i] <= constants[f"pi_u_{i}"]
            model += alphas[i] >= constants[f"alpha_l_{i}"]
            model += alphas[i] <= constants[f"alpha_u_{i}"]

        for i in range(C - 2):
            model += pis[i] <= pis[i + 1]
            model += alphas[i] <= alphas[i + 1]

        status = model.solve()
        if status == 1:
            answers.append(model.objective.value())

    for k in range(C - 1):
        pis = [LpVariable(f"t_{i}") for i in range(C - 1)]
        alphas = [LpVariable(f"q_{i}") for i in range(C - 1)]
        model = LpProblem(name=f"{type}-{k}", sense=LpMaximize)

        model += alphas[k] - pis[k]
        model += pis[k] <= alphas[k]

        for i in range(C - 1):
            model += pis[i] >= constants[f"pi_l_{i}"]
            model += pis[i] <= constants[f"pi_u_{i}"]
            model += alphas[i] >= constants[f"alpha_l_{i}"]
            model += alphas[i] <= constants[f"alpha_u_{i}"]

        for i in range(C - 2):
            model += pis[i] <= pis[i + 1]
            model += alphas[i] <= alphas[i + 1]

        status = model.solve()
        if status == 1:
            answers.append(model.objective.value())

    return sorted(answers)
