# -*- coding: utf-8 -*-
"""
Solve Unique-SVP with Predicate instances.

All functions (which really are instances of a callable class) know about their parameters.  These
parameters can be prepared by calling e.g. ``usvp_pred_bkz_enum_solve.parametersf(A, norm)`` which
will return the required block sizes in this case.

A user probably wants to call ``usvp_pred_solve``.

..  note :: This file assumes that there is at most one vector satisfying the predicate in the
            lattice, i.e. we do early aborts.

"""
# NOTE: This file should not import from the sage namespace,
# i.e. this file is meant to be usable outside SageMath.
from dataclasses import dataclass
import warnings
from math import log, pi, lgamma
import logging

from fpylll import FPLLL, GSO, Enumeration, EnumerationError, Pruning, BKZ
from fpylll.algorithms.bkz2 import BKZReduction as BKZ2
from fpylll.tools.bkz_stats import BKZTreeTracer
from fpylll.util import precision as fplll_precision
from fpylll.tools.bkz_simulator import simulate
from fpylll.util import gaussian_heuristic

from g6k.algorithms.workout import workout
from g6k.siever import SaturationError
from g6k import Siever, SieverParams
from g6k.utils.stats import SieveTreeTracer
from g6k.algorithms.bkz import pump_n_jump_bkz_tour

from utils import SuppressStream


@dataclass
class USVPPredSolverResults:
    """
    All solvers return an instance of this class
    """

    success: bool  # success or not
    ntests: int  # number of calls to the predicate
    b0: float  # norm of the shortest vector in the basis (regardless of predicate)
    solution: tuple = None  # the solution
    cputime: float = None  # cputime spent
    walltime: float = None  # walltime spent
    data: object = None  # any additional data

    def __repr__(self):
        return (
            "USVPPredSolverResults(success={success}, "
            "ntests={ntests}, "
            "b0={b0:.1f}, "
            "solution=({solution}, …), "
            "cputime={cputime:.1f}, "
            "walltime={walltime:.1f}, "
            "data=…)"
        ).format(
            success=int(self.success),
            ntests=self.ntests,
            b0=self.b0,
            cputime=float(self.cputime),
            walltime=float(self.walltime),
            solution=str(self.solution[:5])[1:-1] if self.solution else "None",
        )


STRATEGIES_MAX_DIM = 99


class USVPPredEnum:
    """
    Solve an uSVP with predicate instance with enumeration.

    :param M: FPyLLL ``MatGSO`` object or ``IntegerLattice``
    :param predicate: predicate to evaluate
    :param squared_target_norm: squared norm of target
    :param invalidate_cache: a callable to invalidate caches for the predicate.
    :param target_prob: attempt to achieve this probability of success
    :param preproc_offset: preprocess with block size d - `preproc_offset`
    :param ph: magnitudes are scaled by `2^{ph}` before being considered to avoid overflows
    :param threads: number of threads to use
    :returns: statistics
    :rtype: ``USVPPredSolverResults``

    """

    DEFAULT_TARGET_PROB = 0.9

    @classmethod
    def __call__(
        cls,
        M,
        predicate,
        squared_target_norm,
        invalidate_cache=lambda: None,
        target_prob=None,
        preproc_offset=20,
        ph=0,
        threads=1,
        **kwds
    ):
        preproc_time = None
        ntests = 0

        if target_prob is None:
            target_prob = cls.DEFAULT_TARGET_PROB

        bkz_res = usvp_pred_bkz_enum_solve(
            M, predicate, block_size=min(STRATEGIES_MAX_DIM, M.d), invalidate_cache=invalidate_cache, threads=threads
        )

        if bkz_res.success:  # this might be enough
            return bkz_res

        FPLLL.set_threads(threads)

        M.update_gso()
        bkz = BKZ2(M)
        tracer = BKZTreeTracer(bkz, root_label="enum_pred", start_clocks=True)

        remaining_probability, rerandomize, found, solution = (1.0, False, False, None)

        while remaining_probability > 1.0 - target_prob:
            invalidate_cache()

            with tracer.context("preprocessing"):
                if rerandomize:
                    with tracer.context("randomization"):
                        bkz.randomize_block(0, M.d, tracer=tracer, density=3)
                with tracer.context("reduction"):
                    with tracer.context("lll"):
                        bkz.lll_obj()
                    for _ in range(4):
                        bkz.tour(
                            BKZ.EasyParam(min(max(M.d - preproc_offset, 2), STRATEGIES_MAX_DIM), flags=BKZ.GH_BND),
                            tracer=tracer,
                        )

            if preproc_time is None:
                preproc_time = float(tracer.trace.child("preprocessing")["cputime"])

            with tracer.context("check"):
                for v in M.B:
                    ntests += 1
                    if predicate(v, standard_basis=True):
                        found = True
                        solution = tuple([int(v_) for v_ in v])
                        break

            if found:
                break

            with tracer.context("pruner"):
                preproc_cost = threads * preproc_time * 2 * 10 ** 9 / 100  # 100 cycles per node
                with SuppressStream():
                    r = []
                    for i in range(M.d):
                        r_, exp = M.get_r_exp(i, i)
                        r.append(r_ * 2 ** (exp - ph))
                    (cost, prob), coeffs = cls.pruning_coefficients(
                        squared_target_norm / 2 ** ph, r, preproc_cost, target_prob=target_prob
                    )

            def callbackf(v):
                nonlocal ntests
                ntests += 1
                return predicate(v, standard_basis=False)

            enum_obj = Enumeration(M, callbackf=callbackf)
            with tracer.context("enumeration", enum_obj=enum_obj, probability=prob, full=True):
                try:
                    solutions = enum_obj.enumerate(0, M.d, squared_target_norm / 2 ** ph, ph, pruning=coeffs)
                    _, v = solutions[0]
                    found = True
                    solution = tuple([int(v_) for v_ in M.B.multiply_left(v)])
                    break
                except EnumerationError:
                    pass

            rerandomize = True
            remaining_probability *= 1 - prob

        tracer.exit()
        FPLLL.set_threads(1)

        b0, b0e = bkz.M.get_r_exp(0, 0)

        return USVPPredSolverResults(
            success=found,
            solution=solution,
            ntests=ntests + bkz_res.ntests,
            b0=b0 ** (0.5) * 2 ** (b0e / 2.0),
            cputime=tracer.trace.data["cputime"] + bkz_res.cputime,
            walltime=tracer.trace.data["walltime"] + bkz_res.walltime,
            data=tracer.trace,
        )

    @classmethod
    def pruning_coefficients(cls, squared_target_norm, r, preproc_cost, target_prob=None, precision=212):
        """
        :param squared_target_norm: squared enumeration squared_target_norm
        :param r: basis profile
        :param preproc_cost: preprocessing time in enumeration nodes
        :param target_prob: target probability of success

        """
        if target_prob is None:
            target_prob = cls.DEFAULT_TARGET_PROB

        with fplll_precision(precision):
            for prob, flags in (
                (target_prob, Pruning.GRADIENT | Pruning.HALF),
                (target_prob * 0.9, Pruning.GRADIENT | Pruning.HALF),
                (target_prob * 0.8, Pruning.GRADIENT | Pruning.HALF),
                (target_prob * 0.7, Pruning.GRADIENT | Pruning.HALF),
                (target_prob * 0.6, Pruning.GRADIENT | Pruning.HALF),
                # (target_prob, Pruning.HALF),
            ):
                try:
                    pruner = Pruning.Pruner(
                        squared_target_norm, preproc_cost, [r], target=prob, float_type="mpfr", flags=flags
                    )
                    coeffs = pruner.optimize_coefficients([1.0] * len(r))
                    cost = pruner.repeated_enum_cost(coeffs)
                    return ((preproc_cost + cost, pruner.measure_metric(coeffs)), coeffs)
                except RuntimeError:
                    pass
            else:
                raise RuntimeError("Pruning failed.")

    @classmethod
    def estimate(cls, M, squared_target_norm, target_prob=None):
        """
        :param M: either a GSO object or a tuple containing the ln of the squared volume and the dimension
        :param squared_target_norm: the squared norm of the embedded target vector.
        :returns: cost in CPU cycles, None

        """

        cost, data = usvp_pred_bkz_enum_solve.estimate(M, squared_target_norm)
        if cost:
            return cost, data

        if target_prob is None:
            target_prob = cls.DEFAULT_TARGET_PROB
        try:
            (log_vol, d) = M
            log_vol = log_vol / log(2.0)
        except TypeError:
            try:
                M.update_gso()
            except AttributeError:
                M = GSO.Mat(M)
                M.update_gso()
            d = M.d
            log_vol = M.get_log_det(0, d) / log(2.0)

        preproc_cost = 8 * float(d ** 3)
        preproc_d = max(d - 20, 2)

        for i in range(d):
            d_ = min(preproc_d, d - i)
            if d_ > 30:
                preproc_cost += 8 * float(2 ** (0.1839 * d_ * log(d_, 2) - 0.995 * d_ + 16.25))

        nf = round(log_vol * (1 / d))
        log_vol -= d * nf  # handle rounding errors
        squared_target_norm /= 2 ** nf

        r = [1.0219 ** (2 * (d - 2 * i - 1)) * 2 ** (log_vol * (1 / d)) for i in range(d)]

        from fpylll.tools.bkz_simulator import simulate

        r, _ = simulate(r, BKZ.EasyParam(preproc_d))

        (cost, prob), _ = cls.pruning_coefficients(squared_target_norm, r, preproc_cost, target_prob=target_prob)
        return int(round(cost * 64)), None

    @classmethod
    def parametersf(cls, M, squared_target_norm):
        ph = M.get_r_exp(M.d - 1, M.d - 1)[1]
        # use integers to preserve precision, squared_target_norm might not be a float
        return {"squared_target_norm": 101 * (squared_target_norm / 100), "ph": ph}


usvp_pred_enum_solve = USVPPredEnum()


class USVPPredBKZEnum:
    """
    Solve an uSVP with predicate instance with BKZ+enumeration.

    :param M: FPyLLL ``MatGSO`` object or ``IntegerLattice``
    :param predicate: predicate to evaluate
    :param block_size: BKZ block size
    :param invalidate_cache: a callable to invalidate caches for the predicate.
    :param max_loops: maximum number of BKZ tours
    :param threads: number of threads to use
    :returns: statistics
    :rtype: ``USVPPredSolverResults``

    """

    @classmethod
    def __call__(cls, M, predicate, block_size, invalidate_cache=lambda: None, max_loops=8, threads=1, **kwds):
        bkz = BKZ2(M)

        if block_size > STRATEGIES_MAX_DIM:
            warnings.warn("reducing block size to {max}".format(max=STRATEGIES_MAX_DIM))
            block_size = STRATEGIES_MAX_DIM

        FPLLL.set_threads(threads)
        params = BKZ.EasyParam(block_size=block_size, **kwds)
        auto_abort = BKZ.AutoAbort(M, M.d)
        tracer = BKZTreeTracer(bkz, root_label="bkz_enum", start_clocks=True)
        found, ntests, solution = False, 0, None
        for tour in range(max_loops):
            bkz.tour(params)

            if auto_abort.test_abort():
                break

            invalidate_cache()

            with tracer.context("check"):
                for i, v in enumerate(bkz.M.B):
                    ntests += 1
                    if predicate(v, standard_basis=True):
                        found = True
                        solution = tuple([int(v_) for v_ in v])
                        break
            if found:
                break

        FPLLL.set_threads(1)

        tracer.exit()

        b0, b0e = bkz.M.get_r_exp(0, 0)

        return USVPPredSolverResults(
            success=found,
            solution=solution,
            ntests=ntests,
            b0=b0 ** (0.5) * 2 ** (b0e / 2.0),
            cputime=tracer.trace.data["cputime"],
            walltime=tracer.trace.data["walltime"],
            data=tracer.trace,
        )

    @classmethod
    def estimate(cls, M, squared_target_norm, max_loops=8):
        """
        :param M: either a GSO object or a tuple containing the ln of the squared volume and the dimension
        :param squared_target_norm: the squared norm of the embedded target vector.
        :returns: cost in CPU cycles, block size

        """
        try:
            (log_vol, d) = M
            log_vol = log_vol / log(2.0)
        except TypeError:
            try:
                M.update_gso()
            except AttributeError:
                M = GSO.Mat(M)
                M.update_gso()
            d = M.d
            log_vol = M.get_log_det(0, d) / log(2.0)

        nf = round(log_vol * (1 / d))
        log_vol -= d * nf  # handle rounding errors
        squared_target_norm /= 2 ** nf

        lgh = lgamma(1 + d / 2.0) * (2.0 / d) - log(pi) + log_vol * log(2.0) * (1.0 / d)

        if log(squared_target_norm) > lgh:
            return None, False

        r = [1.0219 ** (2 * (d - 2 * i - 1)) * 2 ** (log_vol * (1 / d)) for i in range(d)]

        found, cost = False, None
        for beta in range(3, d + 1)[::-1]:
            rr = simulate(list(r), BKZ.EasyParam(beta, max_loops=8))[0]
            if squared_target_norm / d * beta < rr[-beta]:
                found = beta
            else:
                break
        if found:
            cost = max_loops * d * float(2 ** (0.1839 * found * log(found, 2.0) - 0.995 * found + 16.25))
            return int(round(cost) * 64), found
        else:
            return None, False

    @classmethod
    def parametersf(cls, M, squared_target_norm):
        block_size = cls.estimate(M, squared_target_norm)[1]
        if not block_size:
            block_size = M.d
        return {"block_size": block_size}


usvp_pred_bkz_enum_solve = USVPPredBKZEnum()


class USVPPredSieve:
    """
    Solve an uSVP with predicate instance with sieving.

    :param M: FPyLLL ``MatGSO`` object or ``IntegerLattice``
    :param predicate: predicate to evaluate
    :param invalidate_cache: a callable to invalidate caches for the predicate.
    :param preproc_offset: preprocess with block size d - `preproc_offset`, preprocessing is disables when 0.
    :param threads: number of threads to use
    :returns: statistics
    :rtype: ``USVPPredSolverResults``

    """

    @classmethod
    def __call__(cls, M, predicate, invalidate_cache=lambda: None, preproc_offset=20, threads=1, **kwds):
        if preproc_offset and M.d >= 40:
            bkz_res = usvp_pred_bkz_sieve_solve(
                M,
                predicate,
                block_size=max(M.d - preproc_offset, 2),
                max_loops=8,
                threads=threads,
                invalidate_cache=invalidate_cache,
            )

            ntests = bkz_res.ntests
            if bkz_res.success:  # this might be enough
                return bkz_res
        else:
            bkz_res = None
            ntests = 0

        params = SieverParams(reserved_n=M.d, otf_lift=False, threads=threads)
        g6k = Siever(M, params)

        tracer = SieveTreeTracer(g6k, root_label="sieve", start_clocks=True)
        workout(g6k, tracer, 0, M.d, dim4free_min=0, dim4free_dec=15)

        invalidate_cache()

        found, solution = False, None
        with tracer.context("check"):  # check if the workout solved it for us
            for i in range(g6k.M.d):
                ntests += 1
                if predicate(g6k.M.B[i], standard_basis=True):
                    found = True
                    solution = tuple([int(v_) for v_ in g6k.M.B[i]])
                    break

        if found:
            tracer.exit()

            b0, b0e = M.get_r_exp(0, 0)

            return USVPPredSolverResults(
                success=found,
                ntests=ntests,
                solution=solution,
                b0=b0 ** (0.5) * 2 ** (b0e / 2.0),
                cputime=tracer.trace.data["cputime"],
                walltime=tracer.trace.data["walltime"],
                data=tracer.trace,
            )

        with tracer.context("sieve"):
            try:
                g6k()
            except SaturationError:
                pass

        while g6k.l:
            g6k.extend_left()
            with tracer.context("sieve"):
                try:
                    g6k()
                except SaturationError:
                    pass

        # fill the database
        with g6k.temp_params(**kwds):
            g6k()

        invalidate_cache()

        with tracer.context("check"):
            for i in range(g6k.M.d):
                ntests += 1
                if predicate(g6k.M.B[i], standard_basis=True):
                    found = True
                    solution = tuple([int(v_) for v_ in g6k.M.B[i]])
                    break

            if not found:
                for v in g6k.itervalues():
                    ntests += 1
                    if predicate(v, standard_basis=False):
                        found = True
                        solution = tuple([int(v_) for v_ in g6k.M.B.multiply_left(v)])
                        break
        tracer.exit()

        cputime = tracer.trace.data["cputime"] + bkz_res.cputime if bkz_res else 0
        walltime = tracer.trace.data["walltime"] + bkz_res.walltime if bkz_res else 0

        b0, b0e = M.get_r_exp(0, 0)

        return USVPPredSolverResults(
            success=found,
            ntests=ntests,
            solution=solution,
            b0=b0 ** (0.5) * 2 ** (b0e / 2.0),
            cputime=cputime,
            walltime=walltime,
            data=tracer.trace,
        )

    @classmethod
    def estimate(cls, M, squared_target_norm):
        """
        :param M: either a GSO object or a tuple containing the ln of the squared volume and the dimension
        :param squared_target_norm: the squared norm of the embedded target vector.
        :returns: cost in CPU cycles, None
        """
        try:
            (log_vol, d) = M
            log_vol = log_vol / log(2.0)
        except TypeError:
            try:
                M.update_gso()
            except AttributeError:
                M = GSO.Mat(M)
                M.update_gso()
            d = M.d
            log_vol = M.get_log_det(0, d) / log(2.0)

        nf = round(log_vol * (1 / d))
        log_vol -= d * nf  # handle rounding errors
        squared_target_norm /= 2 ** nf

        lgh = lgamma(1 + d / 2.0) * (2.0 / d) - log(pi) + log_vol * log(2.0) * (1.0 / d)

        if log(squared_target_norm) - lgh > log(1.01 * 4 / 3.0):  # fudge factor
            return None, None

        # NOTE: please refrain from interpreting the function below. It is merely meant as a method
        # to compress the table that follows. It is not to be interpreted as a prediction for large,
        # cryptographic dimensions.

        # sage: attach("sieve_cost.py")
        # sage: runit(range(40, 101, 2), threads=1, jobs=40, tasks=80)
        # d:  48, cputime:     1.79 (0.082*x + 27.785), walltime:     1.79 (0.077*x + 28.027)
        # d:  50, cputime:     4.97 (0.216*x + 21.883), walltime:     4.97 (0.216*x + 21.904)
        # d:  52, cputime:     5.51 (0.280*x + 18.826), walltime:     5.51 (0.280*x + 18.829)
        # d:  54, cputime:     6.07 (0.277*x + 18.828), walltime:     6.08 (0.277*x + 18.827)
        # d:  56, cputime:     7.46 (0.221*x + 21.639), walltime:     7.46 (0.221*x + 21.638)
        # d:  58, cputime:     8.51 (0.100*x + 28.194), walltime:     8.52 (0.100*x + 28.194)
        # d:  60, cputime:    16.77 (0.185*x + 23.568), walltime:    16.77 (0.185*x + 23.569)
        # d:  62, cputime:    19.02 (0.223*x + 21.341), walltime:    19.02 (0.223*x + 21.341)
        # d:  64, cputime:    21.93 (0.214*x + 21.838), walltime:    21.93 (0.214*x + 21.838)
        # d:  66, cputime:    25.90 (0.180*x + 23.856), walltime:    25.91 (0.180*x + 23.856)
        # d:  68, cputime:    31.72 (0.114*x + 28.074), walltime:    31.72 (0.114*x + 28.075)
        # d:  70, cputime:    62.96 (0.199*x + 22.612), walltime:    62.97 (0.199*x + 22.613)
        # d:  72, cputime:    77.03 (0.245*x + 19.491), walltime:    77.04 (0.245*x + 19.491)
        # d:  74, cputime:    97.72 (0.256*x + 18.717), walltime:    97.73 (0.256*x + 18.717)
        # d:  76, cputime:   129.03 (0.234*x + 20.210), walltime:   129.04 (0.234*x + 20.210)
        # d:  78, cputime:   178.14 (0.187*x + 23.709), walltime:   178.16 (0.187*x + 23.710)
        # d:  80, cputime:   324.95 (0.251*x + 18.964), walltime:   325.03 (0.251*x + 18.962)
        # d:  82, cputime:   447.11 (0.286*x + 16.237), walltime:   447.31 (0.286*x + 16.233)
        # d:  84, cputime:   645.36 (0.299*x + 15.202), walltime:   645.64 (0.299*x + 15.196)
        # d:  86, cputime:   938.59 (0.289*x + 15.946), walltime:   939.01 (0.289*x + 15.942)
        # d:  88, cputime:  1419.92 (0.266*x + 17.898), walltime:  1420.55 (0.266*x + 17.897)
        # d:  90, cputime:  2371.75 (0.298*x + 15.242), walltime:  2372.79 (0.298*x + 15.243)
        # d:  92, cputime:  3742.38 (0.320*x + 13.250), walltime:  3744.01 (0.320*x + 13.251)
        # d:  94, cputime:  5984.56 (0.337*x + 11.747), walltime:  5987.23 (0.337*x + 11.747)
        # d:  96, cputime:  9273.02 (0.337*x + 11.704), walltime:  9276.95 (0.337*x + 11.705)
        # d:  98, cputime: 14675.40 (0.328*x + 12.558), walltime: 14681.85 (0.328*x + 12.559)
        # d: 100, cputime: 24131.73 (0.334*x + 12.071), walltime: 24142.40 (0.334*x + 12.072)

        # sage: runit(range(102, 121, 2), threads=40, jobs=1, tasks=8)
        # d: 102, cputime: 50014.29 (), walltime:  1601.63 ()
        # d: 104, cputime: 81504.44 (), walltime:  2392.41 ()
        # d: 106, cputime: 131605.79 (), walltime:  3653.74 ()
        # d: 108, cputime: 217125.04 (), walltime:  5807.46 ()
        # d: 110, cputime: 359440.48 (0.355*x + 10.268), walltime:  9509.48 (0.321*x + 8.761)
        # d: 112, cputime: 576814.24 (0.355*x + 10.308), walltime: 14967.10 (0.334*x + 7.406)
        # d: 114, cputime: 940493.32 (0.354*x + 10.370), walltime: 24090.69 (0.340*x + 6.650)
        # d: 116, cputime: 1542007.36 (0.352*x + 10.595), walltime: 39181.72 (0.342*x + 6.421)
        # d: 118, cputime: 2509762.46 (0.351*x + 10.699), walltime: 63480.77 (0.343*x + 6.331)
        # d: 120, cputime: 4162026.47 (0.356*x + 10.168), walltime: 105040.10 (0.351*x + 5.445)

        # l = [(x,log(y*2*10**9,2)) for x,y in l]
        # var("x,a,b,c")
        # f = a*x + b*log(x,2) + c
        # f = f.function(x)
        # g = f.subs(find_fit(l[10:], f, solution_dict=True))

        # var("x,a,b,c")
        # f = a*x + b
        # f = f.function(x)
        # g = f.subs(find_fit(l[-10:], f, solution_dict=True))

        # NOTE: We are ignoring the cost of checking the predicate in the database.

        if d <= 90:
            cost = 2 ** float(0.65819 * d - 30.460 * log(d) + 119.91)
        else:
            cost = 2 ** float(0.37495 * d + 8.12)
        return cost, None

    @classmethod
    def parametersf(cls, M, squared_target_norm):
        return {"saturation_ratio": 0.70, "db_size_factor": 3.50}


usvp_pred_sieve_solve = USVPPredSieve()


class USVPPredBKZSieve:
    """
    Solve an uSVP with predicate instance with BKZ+sieving.

    :param M: FPyLLL ``MatGSO`` object or ``IntegerLattice``
    :param predicate: predicate to evaluate
    :param block_size: BKZ block size
    :param invalidate_cache: a callable to invalidate caches for the predicate.
    :param max_loops: maximum number of BKZ tours
    :param threads: number of threads to use
    :returns: statistics
    :rtype: ``USVPPredSolverResults``

    """

    def __call__(cls, M, predicate, block_size, invalidate_cache=lambda: None, threads=1, max_loops=8, **kwds):
        params = SieverParams(threads=threads)
        g6k = Siever(M, params)
        tracer = SieveTreeTracer(g6k, root_label="bkz-sieve", start_clocks=True)
        for b in range(20, block_size + 1, 10):
            pump_n_jump_bkz_tour(g6k, tracer, b, pump_params={"down_sieve": True})

        auto_abort = BKZ.AutoAbort(M, M.d)
        found, ntests, solution = False, 0, None
        for tour in range(max_loops):
            pump_n_jump_bkz_tour(g6k, tracer, block_size, pump_params={"down_sieve": True})

            invalidate_cache()

            if auto_abort.test_abort():
                break

            with tracer.context("check"):
                for i, v in enumerate(M.B):
                    ntests += 1
                    if predicate(v, standard_basis=True):
                        solution = tuple([int(v_) for v_ in v])
                        found = True
                        break
                if found:
                    break

        tracer.exit()

        b0, b0e = M.get_r_exp(0, 0)

        return USVPPredSolverResults(
            success=found,
            ntests=ntests,
            solution=solution,
            b0=b0 ** (0.5) * 2 ** (b0e / 2.0),
            cputime=tracer.trace.data["cputime"],
            walltime=tracer.trace.data["walltime"],
            data=tracer.trace,
        )

    @classmethod
    def estimate(cls, M, squared_target_norm, max_loops=8):
        """

        :param M:
        :param squared_target_norm:
        :param target_prob:
        :returns: cost in CPU cycles

        """
        try:
            (_, d) = M
        except TypeError:
            try:
                M.update_gso()
            except AttributeError:
                M = GSO.Mat(M)
                M.update_gso()
            d = M.d

        _, block_size = USVPPredBKZEnum.estimate(M, squared_target_norm)
        if block_size:
            # TODO: this seems way too much
            cost = max_loops * d * 2 ** float(0.38191949470057696 * block_size - 32.71092701524247) * 3600 * 2 * 10 ** 9
            return cost, block_size
        else:
            return None, False

    @classmethod
    def parametersf(cls, M, squared_target_norm):
        block_size = USVPPredBKZEnum.estimate(M, squared_target_norm)[1]
        if not block_size:
            block_size = M.d
        return {"block_size": block_size}


usvp_pred_bkz_sieve_solve = USVPPredBKZSieve()

solvers = {
    "bkz-enum": usvp_pred_bkz_enum_solve,
    "bkz-sieve": usvp_pred_bkz_sieve_solve,
    "enum_pred": usvp_pred_enum_solve,
    "sieve_pred": usvp_pred_sieve_solve,
}


def usvp_pred_solve(A, predicate, squared_target_norm, invalidate_cache=lambda: None, solver=None, **kwds):
    """
    Solve uSVP with predicate.

    Given a USVP instance ``A`` with ``predicate`` and a target of ``squared_target_norm`` solve
    this intance using ``solver``.

    :param A: An ``IntegerMatrix`` or a ``MatGSO`` object
    :param predicate: a predicate (this will inject ``M`` into its global namespace)
    :param squared_target_norm: the squared norm of the target
    :param invalidate_cache: a callable to invalidate caches for the predicate.
    :param solver: uSVP with predicate solver to use.

    """
    from g6k import Siever

    try:
        solver = solvers[solver]
    except KeyError:
        pass

    try:
        A.update_gso()
        M = A
    except AttributeError:
        M = Siever.MatGSO(A)
        M.update_gso()

    predicate.__globals__["M"] = M

    if solver is None:
        cost, block_size = usvp_pred_bkz_enum_solve.estimate(M, squared_target_norm)
        if cost:  # HACK
            if block_size >= 70:
                solver_name = "bkz-sieve"
            else:
                solver_name = "bkz-enum"
        else:
            gh = gaussian_heuristic(M.r())
            if M.d < 40 or squared_target_norm / gh > 4 / 3.0:
                solver_name = "enum_pred"
            else:
                solver_name = "sieve_pred"
        solver = solvers[solver_name]
        logging.debug("% solving with {solver_name}".format(solver_name=solver_name))

    aux_kwds = kwds
    kwds = solver.parametersf(M, squared_target_norm)
    kwds.update(aux_kwds)

    logging.debug("% solving with {kwds}".format(kwds=kwds))

    return solver(M, predicate, invalidate_cache=invalidate_cache, **kwds)


def usvp_pred_solve_scale(
    A, predicate, squared_target_norm, invalidate_cache=lambda: None, solver=None, scale_factor=1200, **kwds
):
    """
    Solve uSVP with predicate, on failure increase target norm and try again

    Given a USVP instance ``A`` with ``predicate`` and a target of ``squared_target_norm`` solve
    this intance using ``solver``, increasing the search radius on failure.

    :param A: An ``IntegerMatrix`` or a ``MatGSO`` object
    :param predicate: a predicate (this will inject ``M`` into its global namespace)
    :param squared_target_norm: the squared norm of the target
    :param invalidate_cache: a callable to invalidate caches for the predicate.
    :param solver: uSVP with predicate solver to use, it is probably a bad idea to set this
    :param scale_factor: on failure ``squared_target_norm`` is scaled by ``scale_factor/1000``

    """
    # TODO: accumulate costs
    ntests, cputime, walltime = 0, 0, 0
    while True:
        ret = usvp_pred_solve(
            A=A,
            predicate=predicate,
            squared_target_norm=squared_target_norm,
            invalidate_cache=invalidate_cache,
            solver=solver,
            **kwds
        )
        if ret.success:
            break
        else:
            ntests += ret.ntests
            cputime += ret.cputime
            walltime += ret.walltime
            squared_target_norm = (squared_target_norm * scale_factor) / 1000

    return USVPPredSolverResults(
        success=ret.success,
        ntests=ret.ntests + ntests,
        solution=ret.solution,
        b0=ret.b0,
        cputime=ret.cputime + cputime,
        walltime=ret.walltime + walltime,
        data=ret.data,
    )


def usvp_pred_solve_repeat(
    A, predicate, squared_target_norm, invalidate_cache=lambda: None, solver=None, repeat=10, **kwds
):
    """
    Solve uSVP with predicate, repeatedly

    Given a USVP instance ``A`` with ``predicate`` and a target of ``squared_target_norm`` solve
    this intance using ``solver`` up to ``repeat`` times (since our algorithms are probabilistic).

    :param A: An ``IntegerMatrix`` or a ``MatGSO`` object
    :param predicate: a predicate (this will inject ``M`` into its global namespace)
    :param squared_target_norm: the squared norm of the target
    :param invalidate_cache: a callable to invalidate caches for the predicate.
    :param solver: uSVP with predicate solver to use, it is probably a bad idea to set this
    :param repeat: try this many times

    """
    ntests, cputime, walltime = 0, 0, 0
    for _ in range(repeat):
        ret = usvp_pred_solve(
            A=A,
            predicate=predicate,
            squared_target_norm=squared_target_norm,
            invalidate_cache=invalidate_cache,
            solver=solver,
            **kwds
        )
        if ret.success:
            break
        else:
            ntests += ret.ntests
            cputime += ret.cputime
            walltime += ret.walltime

    return USVPPredSolverResults(
        success=ret.success,
        ntests=ret.ntests + ntests,
        solution=ret.solution,
        b0=ret.b0,
        cputime=ret.cputime + cputime,
        walltime=ret.walltime + walltime,
        data=ret.data,
    )


flavors = {"plain": usvp_pred_solve, "repeat": usvp_pred_solve_repeat, "scale": usvp_pred_solve_scale}
