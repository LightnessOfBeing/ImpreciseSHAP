from itertools import chain, combinations
from typing import Iterable, List


def powerset(iterable: Iterable[str]) -> List[List[str]]:
    s = list(iterable)
    return list(
        map(
            lambda x: list(x),
            chain.from_iterable(list(combinations(s, r)) for r in range(len(s) + 1)),
        )
    )
