# -*- coding: utf-8 -*-


def test_ecdsa():
    from ecdsa_hnp import ECDSA

    for curve in ECDSA.supported_curves():
        ECDSA(curve)


def test_ecdsa_benchmark():
    from ecdsa_hnp import benchmark
    from usvp import solvers

    for solver in solvers:
        benchmark(nlen=256, m=2, klen=128, tasks=1, algorithm=solver, seed=0)
