"""
Sieve on the top-most bits of the matrix.
"""

from g6k import Siever, SieverParams
from g6k.siever import SaturationError
from g6k.utils.stats import SieveTreeTracer


from usvp import USVPPredSolverResults, USVPPredSieve, usvp_pred_bkz_enum_solve


class USVPPredCutNSieve(USVPPredSieve):
    """
    Solve an uSVP with predicate instance with sieving, considering only top most bits of matrix entries.

    :param M: FPyLLL ``MatGSO`` object or ``IntegerLattice``
    :param predicate: predicate to evaluate
    :param invalidate_cache: a callable to invalidate caches for the predicate.
    :param preproc_offset: preprocess with block size d - `preproc_offset`, preprocessing is disables when 0.
    :param threads: number of threads to use
    :returns: statistics
    :rtype: ``USVPPredSolverResults``

    """

    @classmethod
    def __call__(cls, M, predicate, invalidate_cache=lambda: None, preproc_offset=20, threads=1, ph=0, **kwds):
        # TODO bkz_sieve would be neater here
        if preproc_offset and M.d >= 40:
            bkz_res = usvp_pred_bkz_enum_solve(
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

        from fpylll import IntegerMatrix

        # reduce size of entries
        B = IntegerMatrix(M.B.nrows, M.B.ncols)
        for i in range(M.B.nrows):
            for j in range(M.B.ncols):
                B[i, j] = M.B[i, j] // 2**ph

        params = SieverParams(reserved_n=M.d, otf_lift=False, threads=threads)
        g6k = Siever(B, params)
        tracer = SieveTreeTracer(g6k, root_label="sieve", start_clocks=True)

        g6k.initialize_local(0, M.d // 2, M.d)

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

        found, solution = False, None
        with tracer.context("check"):
            for v in g6k.itervalues():  # heuristic: v has very small entries
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
    def parametersf(cls, M, squared_target_norm):

        kwds = USVPPredSieve.parametersf(M, squared_target_norm)
        ph = int(M.B[0, 0]).bit_length()
        for i in range(M.B.nrows):
            for j in range(M.B.ncols):
                if M.B[i, j]:
                    ph = min(int(M.B[i, j]).bit_length(), ph)
        kwds["ph"] = max(ph - 128, 0)
        return kwds


usvp_pred_cut_n_sieve_solve = USVPPredCutNSieve()
