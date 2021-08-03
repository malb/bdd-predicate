"""
Solve the Hidden Number Problem defined by ECDSA with biased nonces (ECDSA-HNP) using lattices.
"""

from sage.all import (
    vector,
    mod,
    Combinations,
    log_gamma,
    exp,
    Integer,
    FiniteField,
    EllipticCurve,
    ZZ,
    lift,
    inverse_mod,
    sqrt,
    log,
    set_random_seed,
    median,
    mean,
    RR,
    RealField,
    pi,
    floor,
    ceil,
    cached_method,
    random,
)

from collections import namedtuple
from multiprocessing import Pool
import binascii
import datetime
import logging
import multiprocessing_logging
import socket

from fpylll import IntegerMatrix, FPLLL, GSO, LLL

from usvp import flavors
from utils import btoi, itob


multiprocessing_logging.install_mp_handler()


class ECDSA(object):
    """
    ECDSA operations are collected in this object.
    """

    @classmethod
    def supported_curves(cls):
        import ecdsa

        curves = {str.lower(str(c)): c for c in ecdsa.curves.curves}
        return curves

    def __init__(self, curve=None, nbits=256):
        curves = self.supported_curves()

        if curve is None:
            if nbits == 160:
                curve = "brainpoolp160r1"
            elif nbits == 192:
                curve = "nist192p"
            elif nbits == 224:
                curve = "nist224p"
            elif nbits == 256:
                curve = "secp256k1"
            elif nbits == 320:
                curve = "brainpoolp320r1"
            elif nbits == 384:
                curve = "nist384p"
            elif nbits == 521:
                curve = "nist521p"
            else:
                raise NotImplementedError("nlen={nlen} is not implemented".format(nlen=nbits))

        if str.lower(curve) in curves.keys():
            self.curve = curves[curve]
            self.baselen = self.curve.baselen
            self.nbits = self.curve.order.bit_length()
            self.G = self.curve.generator
            self.n = Integer(self.G.order())
            self.F = FiniteField(self.curve.curve.p())
            self.C = EllipticCurve([self.F(self.curve.curve.a()), self.F(self.curve.curve.b())])
            self.GG = self.C([self.G.x(), self.G.y()])
        else:
            raise NotImplementedError("curve={curve} is not implemented".format(curve=curve))

    def sign(self, h, sk, klen=256, return_k=False):
        """
        Sign ``h`` and signing key ``sk``

        :param h: "hash"
        :param sk: signing key
        :param klen: number of bits in the nonce.
        :param return_k:

        """
        d = btoi(sk.to_string())
        hi = btoi(h)
        k = ZZ.random_element(2 ** klen)
        r = Integer((self.GG * k).xy()[0])
        s = lift(inverse_mod(k, self.n) * mod(hi + d * r, self.n))
        sig = itob(r, self.baselen) + itob(s, self.baselen)
        if return_k:
            return k, sig
        return sig

    def sample(self, m=2, klen_list=None, seed=None, errors=0.0):
        """
        Sample `m` leaky signatures.

        :param m:
        :param klen_list:
        :param seed:
        :param errors: fraction of signatures that are 1 bit longer than ``klen_list`` specifies

        """
        if klen_list is None:
            klen_list = [128] * m
        import ecdsa as ecdsam
        from ecdsa.util import PRNG

        if seed is not None:
            rng = PRNG(seed)
        else:
            rng = None
        sk = ecdsam.SigningKey.generate(curve=self.curve, entropy=rng)
        d = btoi(sk.to_string())
        vk = sk.get_verifying_key()
        lines = []
        k_list = []
        for i in range(m):
            h = ZZ.random_element(2 ** self.nbits)
            hb = itob(h, self.baselen)
            if errors > 0 and random() < errors:
                k, sss = self.sign(hb, sk, klen=klen_list[i] + 1, return_k=True)
            else:
                k, sss = self.sign(hb, sk, klen=klen_list[i], return_k=True)
            k_list.append(k)
            lines.append("%s %s %s %s" % (str(klen_list[i]), bytes.hex(hb), bytes.hex(sss), bytes.hex(vk.to_string())))
        return lines, k_list, d


class ECDSASolver(object):
    """
    Solve ECDSA with biased nonces.
    """

    def __init__(self, ecdsa, lines, m, d=None, threads=1):
        """

        :param ecdsa: ECDSA object
        :param lines: TODO
        :param m: number of samples
        :param d: dimension of the lattice (default: `m+1`)
        :param threads: number of threads to use

        """
        from ecdsa import VerifyingKey

        self.ecdsa = ecdsa
        self.klen_list = []
        self.s_list = []
        self.r_list = []
        self.h_list = []
        self.m = 0
        self.vk = ""
        self.d = m + 1 if d is None else d
        self.threads = threads

        for line in lines:
            klen, h, sig, key = line.strip().split()
            self.klen_list.append(int(klen))
            self.h_list.append(int(h, 16))
            if not self.vk:
                self.pubx = btoi(binascii.unhexlify(key[: self.ecdsa.baselen * 2]))
                self.puby = btoi(binascii.unhexlify(key[self.ecdsa.baselen * 2 :]))
                vk = VerifyingKey.from_string(
                    itob(self.pubx, self.ecdsa.baselen) + itob(self.puby, self.ecdsa.baselen), curve=self.ecdsa.curve
                )
                self.vk = vk
            r = sig[: 2 * self.ecdsa.baselen]
            self.r_list.append(int(r, 16))
            s = sig[2 * self.ecdsa.baselen :]
            self.s_list.append(int(s, 16))
            assert self.vk.verify_digest(binascii.unhexlify(r + s), binascii.unhexlify(h))
            self.m += 1
            if m != 0 and self.m >= m:
                break

        # TODO: we probably don't want to walk through this in order but randomised
        self.indices = Combinations(range(self.m), self.d - 1)
        self.nbases = 0

    def gen_lattice(self, d=None):
        """FIXME! briefly describe function

        :param d:

        """

        try:
            I = self.indices[self.nbases]  # noqa
            self.nbases += 1
        except ValueError:
            raise StopIteration("No more bases to sample.")
        p = self.ecdsa.n
        # w = 2 ** (self.klen - 1)
        w_list = [2 ** (klen - 1) for klen in self.klen_list]

        r_list = [self.r_list[i] for i in I]
        s_list = [self.s_list[i] for i in I]
        h_list = [self.h_list[i] for i in I]

        rm = r_list[-1]
        sm = s_list[-1]
        hm = h_list[-1]
        wm = w_list[-1]
        a_list = [
            lift(
                wi
                - mod(r, p) * inverse_mod(s, p) * inverse_mod(rm, p) * mod(sm, p) * wm
                - inverse_mod(s, p) * mod(h, p)
                + mod(r, p) * inverse_mod(s, p) * mod(hm, p) * inverse_mod(rm, p)
            )
            for wi, h, r, s in zip(w_list[:-1], h_list[:-1], r_list[:-1], s_list[:-1])
        ]
        t_list = [
            -lift(mod(r, p) * inverse_mod(s, p) * inverse_mod(rm, p) * sm) for r, s in zip(r_list[:-1], s_list[:-1])
        ]

        d = self.d
        A = IntegerMatrix(d, d)

        f_list = [Integer(max(w_list) / w) for w in w_list]

        for i in range(d - 2):
            A[i, i] = p * f_list[i]

        for i in range(d - 2):
            A[d - 2, i] = t_list[i] * f_list[i]
        A[d - 2, d - 2] = f_list[-1]

        for i in range(d - 2):
            A[d - 1, i] = a_list[i] * f_list[i]
        A[d - 1, d - 1] = max(w_list)

        if self.ecdsa.nbits > 384:
            M = GSO.Mat(
                A,
                U=IntegerMatrix.identity(A.nrows, int_type=A.int_type),
                UinvT=IntegerMatrix.identity(A.nrows, int_type=A.int_type),
                float_type="ld",
                flags=GSO.ROW_EXPO,
            )
        else:
            M = GSO.Mat(
                A,
                U=IntegerMatrix.identity(A.nrows, int_type=A.int_type),
                UinvT=IntegerMatrix.identity(A.nrows, int_type=A.int_type),
                flags=GSO.ROW_EXPO,
            )
        M.update_gso()
        return M

    def test_r(self, k, r, w):
        """FIXME! briefly describe function

        :param k:
        :param r:
        :param w:

        """
        for kk in [-k + w, k + w]:
            testpub = self.ecdsa.GG * kk
            if testpub.xy()[0] == r:
                return True
        return False

    def recover_key(self, solution_vector):
        w = 2 ** (self.klen_list[0] - 1)
        f = Integer((2 ** (max(self.klen_list) - 1)) / w)

        def test_key(k):
            if (k * self.ecdsa.GG).xy()[0] == self.r_list[0]:
                d = Integer(
                    mod(inverse_mod(self.r_list[0], self.ecdsa.n) * (k * self.s_list[0] - self.h_list[0]), self.ecdsa.n)
                )
                pubkey = self.ecdsa.GG * d
                if (
                    itob(pubkey.xy()[0], self.ecdsa.baselen) + itob(pubkey.xy()[1], self.ecdsa.baselen)
                    == self.vk.to_string()
                ):
                    return True, d
            return False, None

        result, d = test_key(solution_vector[0] // f + w)
        if result:
            return d
        result, d = test_key(-solution_vector[0] // f + w)
        if result:
            return d
        return False

    @cached_method
    def _data_for_test(self, M=None):
        """
        Return precomputed data used in predicate tests.
        """
        if M is None:
            M = self.M

        w = 2 ** (self.klen_list[0] - 1)
        f = Integer((2 ** (max(self.klen_list) - 1)) / w)
        G_powers = {}
        for row in range(M.B.nrows):
            G_powers[Integer(M.B[row][0] / f)] = Integer(M.B[row][0] / f) * self.ecdsa.GG
        G_powers[w] = w * self.ecdsa.GG

        A0 = tuple([Integer(M.B[i][0] / f) for i in range(M.B.nrows)])
        A1 = tuple([M.B[i][-1] for i in range(M.B.nrows)])
        return G_powers, A0, A1

    @classmethod
    def volf(cls, m, p, klen_list, prec=53):
        """
        Lattice volume.

        :param m: number of samples
        :param p: ECDSA modulus
        :param klen_list: list of lengths of key to recover
        :param prec: precision to use

        """
        w = 2 ** (max(klen_list) - 1)
        RR = RealField(prec)
        f_list = [Integer(w / (2 ** (klen - 1))) for klen in klen_list]
        return RR(exp(log(p) * (m - 1) + sum(map(log, f_list)) + log(w)))

    @classmethod
    def ghf(cls, m, p, klen_list, prec=53):
        """
        Estimate norm of shortest vector according to Gaussian Heuristic.

        :param m: number of samples
        :param p: ECDSA modulus
        :param klen_list: list of lengths of key to recover
        :param prec: precision to use

        """
        # NOTE: The Gaussian Heuristic does not hold in small dimensions
        w = 2 ** (max(klen_list) - 1)
        RR = RealField(prec)
        w = RR(w)
        f_list = [Integer(w / (2 ** (klen - 1))) for klen in klen_list]
        d = m + 1
        log_vol = log(p) * (m - 1) + sum(map(log, f_list)) + log(w)
        lgh = log_gamma(1 + d / 2.0) * (1.0 / d) - log(sqrt(pi)) + log_vol * (1.0 / d)
        return RR(exp(lgh))

    @classmethod
    def evf(cls, m, max_klen, prec=53):
        """
        Estimate norm of target vector.

        :param m: number of samples
        :param klen: length of key to recover
        :param prec: precision to use

        """
        w = 2 ** (max_klen - 1)
        RR = RealField(prec)
        w = RR(w)
        return RR(sqrt(m * (w ** 2 / 3 + 1 / RR(6)) + w ** 2))

    @classmethod
    def mvf(cls, m, max_klen, prec=53):
        """
        Maximal norm of target vector.

        :param m: number of samples
        :param max_klen: length of key to recover
        :param prec: precision to use


        """
        w = 2 ** (max_klen - 1)
        RR = RealField(prec)
        w = RR(w)
        d = m + 1
        return RR(sqrt(d) * w)

    def __call__(self, solver=None, flavor="plain", worst_case=False, sample=True, **kwds):
        """
        Solve the HNP instance.

        :param solver: a uSVP with predicate solver or ``None`` for letting ``usvp_pred_solve`` decide.
        :param sample: if ``True`` a fresh basis is sampled
        :param worst_case: if ``True`` the target norm is chosen to match the maximum of the target, this will be slow.

        """
        if sample:
            self.M = self.gen_lattice()

        tau = max([2 ** (klen - 1) for klen in self.klen_list])

        def predicate(v, standard_basis=True):
            G_powers, A0, A1 = self._data_for_test()
            w = 2 ** (self.klen_list[0] - 1)
            f = Integer((2 ** (max(self.klen_list) - 1)) / w)

            if standard_basis:
                nz = v[-1]
            else:
                nz = sum(round(v[i]) * A1[i] for i in range(len(A1)))  # the last coefficient must be non-zero

            if abs(nz) != tau:
                return False

            if standard_basis:
                kG = G_powers[v[0] // f]
            else:
                kG = sum(round(v[i]) * G_powers[A0[i]] for i in range(len(A0)))

            r = self.r_list[0]
            if (kG + G_powers[w]).xy()[0] == r:
                return True
            elif (-kG + G_powers[w]).xy()[0] == r:
                return True
            else:
                return False

        def invalidate_cache():
            self._data_for_test.clear_cache()

        if worst_case:
            target_norm = self.mvf(self.m, max(self.klen_list), prec=self.ecdsa.nbits // 2)
        else:
            target_norm = self.evf(self.m, max(self.klen_list), prec=self.ecdsa.nbits // 2)

        LLL.Reduction(self.M)()
        invalidate_cache()

        res = flavors[flavor](
            self.M,
            predicate,
            squared_target_norm=target_norm ** 2,
            invalidate_cache=invalidate_cache,
            threads=self.threads,
            solver=solver,
            **kwds
        )

        if res.success:
            key = self.recover_key(res.solution)
        else:
            key = False

        return key, res


def make_klen_list(klen, m):
    if klen in ZZ:
        klen_list = [int(klen)] * m
    else:
        nz = int(round((ceil(klen) - klen) * m))
        klen_list = [floor(klen)] * nz + [ceil(klen)] * (m - nz)
    return klen_list


ComputeKernelParams = namedtuple(
    "ComputeKernelParams",
    ("i", "nlen", "m", "e", "klen_list", "seed", "algorithm", "flavor", "d", "threads", "tag", "params"),
)


def compute_kernel(args):
    if args.seed is not None:
        set_random_seed(args.seed)
        FPLLL.set_random_seed(args.seed)

    ecdsa = ECDSA(nbits=args.nlen)

    lines, k_list, _ = ecdsa.sample(m=args.m, klen_list=args.klen_list, seed=args.seed, errors=args.e)
    w_list = [2 ** (klen - 1) for klen in args.klen_list]
    f_list = [Integer(max(w_list) / wi) for wi in w_list]

    targetvector = vector([(k - w) * f for k, w, f in zip(k_list, w_list, f_list)] + [max(w_list)])

    try:
        solver = ECDSASolver(ecdsa, lines, m=args.m, d=args.d, threads=args.threads)
    except KeyError:
        raise ValueError("Algorithm {alg} unknown".format(alg=args.alg))

    expected_length = solver.evf(args.m, max(args.klen_list), prec=args.nlen // 2)
    gh = solver.ghf(args.m, ecdsa.n, args.klen_list, prec=args.nlen // 2)
    params = args.params if args.params else {}
    key, res = solver(solver=args.algorithm, flavor=args.flavor, **params)

    RR = RealField(args.nlen // 2)
    logging.info(
        (
            "try: {i:3d}, tag: 0x{tag:016x}, success: {success:1d}, "
            "|v|: 2^{v:.2f}, |b[0]|: 2^{b0:.2f}, "
            "|v|/|b[0]|: {b0r:.3f}, "
            "E|v|/|b[0]|: {eb0r:.3f}, "
            "|v|/E|b[0]|: {b0er:.3f}, "
            "cpu: {cpu:10.1f}s, "
            "wall: {wall:10.1f}s, "
            "work: {total:d}"
        ).format(
            i=args.i,
            tag=args.tag,
            success=int(res.success),
            v=float(log(RR(targetvector.norm()), 2)),
            b0=float(log(RR(res.b0), 2)),
            b0r=float(RR(targetvector.norm()) / RR(res.b0)),
            eb0r=float(RR(expected_length) / RR(res.b0)),
            b0er=float(RR(targetvector.norm()) / gh),
            cpu=float(res.cputime),
            wall=float(res.walltime),
            total=res.ntests,
        )
    )

    return key, res, float(targetvector.norm())


def benchmark(
    nlen=256,
    klen=128,
    m=2,
    e=0.0,
    tasks=8,
    algorithm=None,
    flavor="plain",
    d=None,
    jobs=1,
    parallelism=1,
    seed=None,
    solver_params=None,
):
    """

    :param nlen: number of bits in the ECDSA key
    :param klen: number of known bits of the key
    :param m: number of available samples
    :param e: fraction of errors
    :param tasks: number of experiments to run
    :param algorithm: algorithm to use, see ``usvp.solvers``
    :param flavor: higher-level strategy to use, see ``usvp.flavors``
    :param d: lattice dimension (default: `m+1`)
    :param jobs: number of experiments to run in parallel
    :param parallelism: parallelism to use per experiment
    :param seed: randomness seed

    """
    from usvp_prec_hack import usvp_pred_cut_n_sieve_solve
    from usvp import solvers

    if nlen > 384:
        logging.warning("% hotpatching with slower but more numerically stable `usvp_pred_cut_n_sieve_solve`.")
        solvers["sieve_pred"] = usvp_pred_cut_n_sieve_solve

    klen_list = make_klen_list(klen, m)

    tag = ZZ.random_element(x=0, y=2 ** 64)  # we tag all outputs for easier matching

    if seed is None:
        seed = ZZ.random_element(x=0, y=2 ** 64)

    logging.warning(
        (
            "% {t:s} {h:s} 0x{tag:016x} :: nlen: {nlen:3d}, m: {m:2d}, klen: {klen:.3f}, e: {e:.2f}, "
            "alg: {alg:s}, seed: 0x{seed:016x}, params: {params}"
        ).format(
            t=str(datetime.datetime.now()),
            h=socket.gethostname(),
            nlen=nlen,
            e=e,
            m=m,
            klen=float(mean(klen_list)),
            alg=str(algorithm),
            seed=seed,
            tag=tag,
            params=solver_params,
        )
    )

    pool = Pool(jobs)
    J = [
        ComputeKernelParams(
            i=i,
            nlen=nlen,
            m=m,
            e=e,
            klen_list=klen_list,
            seed=seed + i,
            algorithm=algorithm,
            flavor=flavor,
            d=d,
            threads=parallelism,
            params=solver_params,
            tag=tag,
        )
        for i in range(tasks)
    ]

    if jobs > 1:
        r = list(pool.imap_unordered(compute_kernel, J))
    else:
        r = list(map(compute_kernel, J))

    ecdsa = ECDSA(nbits=nlen)

    expected_target = ECDSASolver.evf(m, max(klen_list), prec=nlen // 2)
    expected_b0 = ECDSASolver.ghf(m, ecdsa.n, klen_list, prec=nlen // 2)

    successes = 0
    B0 = []
    eB0 = []
    work = []
    cputime = []
    walltime = []
    for key, res, targetvector_norm in r:
        successes += int(res.success)
        B0.append(float(targetvector_norm / res.b0))
        eB0.append(float(expected_target / res.b0))
        work.append(int(res.ntests))
        cputime.append(float(res.cputime))
        walltime.append(float(res.walltime))

    logging.warning(
        (
            "% {tm:s} {h:s} 0x{tag:016x} ::  sr: {sr:3.0f}%, v/b[0]: {b0ratio:.3f}, "
            "E|v|/|b[0]|: {eb0r:.3f}, E|v|/E|b[0]|: {eveb:.3f}, work: {wk:d}, "
            "t: {t:.1f}s, w: {w:.1f}s"
        ).format(
            tm=str(datetime.datetime.now()),
            h=socket.gethostname(),
            sr=100 * float(successes / tasks),
            b0ratio=median(B0),
            eb0r=median(eB0),
            eveb=float(expected_target / expected_b0),
            wk=int(median(work)),
            t=median(cputime),
            w=median(walltime),
            tag=tag,
        )
    )

    return r


def estimate(nlen=256, m=85, klen=254, skip=None):
    """
    Estimate the cost of solving HNP for an ECDSA with biased nonces instance.

    :param nlen:
    :param m:
    :param klen:
    :param compute:
    :returns:
    :rtype:

    EXAMPLES::

        sage: estimate(256, m=85, klen=254)
        sage: estimate(160, m=85, klen=158)

    """
    from usvp import solvers

    if skip is None:
        skip = []

    ecdsa = ECDSA(nbits=nlen)
    klen_list = make_klen_list(klen, m)
    gh = ECDSASolver.ghf(m, ecdsa.n, klen_list, prec=nlen // 2)
    vol = ECDSASolver.volf(m, ecdsa.n, klen_list, prec=nlen // 2)
    target_norm = ECDSASolver.evf(m, max(klen_list), prec=nlen // 2)

    print(
        ("% {t:s} {h:s}, nlen: {nlen:3d}, m: {m:2d}, klen: {klen:.3f}").format(
            t=str(datetime.datetime.now()), h=socket.gethostname(), nlen=nlen, m=m, klen=float(mean(klen_list))
        )
    )

    print("     E[|b[0]|]: 2^{v:.2f}".format(v=float(RR(log(gh, 2)))))
    print("        E[|v|]: 2^{v:.2f}".format(v=float(RR(log(target_norm, 2)))))
    print("  E[v]/E[b[0]]: %.4f" % float(target_norm / gh))
    print("")

    for solver in solvers:
        if solver in skip:
            continue
        cost, params = solvers[solver].estimate((2 * log(vol), m + 1), target_norm ** 2)
        if cost is None:
            print(" {solver:20s} not applicable".format(solver=solver))
            continue
        else:
            print(
                " {solver:20s} cost: 2^{c:.1f} cycles ≈ {t:12.4f}h, aux data: {params}".format(
                    solver=solver, c=float(log(cost, 2)), t=cost / (2.0 * 10.0 ** 9 * 3600.0), params=params
                )
            )
