#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Command-line interface for solving ECDSA with known nonce bits
"""
import re
import csv
from collections import OrderedDict
from dataclasses import dataclass
from functools import total_ordering

import click
import logging
from multiprocessing_logging import install_mp_handler

from fpylll.tools.bkz_stats import Accumulator

from math import ceil, log
import usvp
from ecdsa_hnp import ECDSA


@click.group()
def ecdsa():
    "Solving ECDSA with known nonce bits using lattice reduction on HNP instances."
    pass


@ecdsa.command()
@click.option("-n", "--nlen", help="Bit length of curve", default=256)
@click.option("-k", "--klen", help="Number of nonce bits", default=128, type=float)
@click.option("-m", "--m", default=2, help="Number of samples considered per instance")
@click.option("-e", "--e", default=0.0, help="Fraction of (input) errors.")
@click.option(
    "-a",
    "--algorithm",
    help="Solving algorithm",
    type=click.Choice(usvp.solvers.keys(), case_sensitive=False),
    multiple=True,
    default=(None,),
)
@click.option(
    "-f",
    "--flavor",
    help="Higher-level strategy",
    type=click.Choice(usvp.flavors.keys(), case_sensitive=False),
    default="plain",
)
@click.option("-t", "--tasks", help="Number of experiments to run", default=8)
@click.option("-j", "--jobs", help="Number of parallel jobs", default=1)
@click.option("-p", "--parallelism", help="Number of threads per job", default=1)
@click.option("-s", "--seed", help="Randomness seed", type=int, default=None)
@click.option(
    "-d", "--dimension", help="Lattice dimension ≤ m+1, default: m+1", type=int, default=None
)
@click.option(
    "-P",
    "--params",
    help=(
        "Key-value pairs passed to the solver;"
        " values are `eval`d;"
        " pass multiple times for passing multiple parameters"
    ),
    type=(str, str),
    multiple=True,
    default=None,
)
@click.option(
    "--loglvl",
    help="Level of verbosity",
    type=click.Choice(["DEBUG", "INFO", "WARNING"], case_sensitive=False),
    default="INFO",
)
def benchmark(
    nlen,
    klen,
    m,
    e,
    algorithm,
    flavor,
    tasks,
    jobs,
    parallelism,
    seed,
    dimension,
    params,
    loglvl="DEBUG",
):
    """
    Generate random instances and report solving statistics.

    Fractional nonce lengths (`-k`) are interpreted as a fraction of larger nonces.  For example
    `252.2` means `0.2⋅m` nonces of size `2^{253}` and `0.8⋅m` of size `2^{252}`.

    """

    logging.basicConfig(level=loglvl, format="%(message)s")
    install_mp_handler()

    from ecdsa_hnp import benchmark as run_hnp
    from sage.all import ZZ

    if klen >= nlen:
        raise ValueError("{klen:.2f} ≥ {nlen}".format(klen=klen, nlen=nlen))

    if params is None:
        params = tuple()
    if params:
        params = dict([(x, eval(y)) for x, y in params])
    else:
        params = {}
    if algorithm == tuple():
        algorithm = (None,)

    if seed is None:
        seed = ZZ.random_element(x=0, y=2**64)

    for alg in algorithm:
        run_hnp(
            nlen=nlen,
            klen=klen,
            m=m,
            e=e,
            tasks=tasks,
            algorithm=alg,
            flavor=flavor,
            d=dimension,
            jobs=jobs,
            parallelism=parallelism,
            seed=seed,
            solver_params=params,
        )


@ecdsa.command()
@click.option(
    "-c",
    "--curve",
    help="Name of curve",
    type=click.Choice(ECDSA.supported_curves().keys(), case_sensitive=False),
    default="secp256k1",
)
@click.option(
    "-m",
    "--m",
    default=None,
    type=int,
    help="Number of samples considered (optional, default: all)",
)
@click.option(
    "-a",
    "--algorithm",
    help="Solving algorithm",
    type=click.Choice(usvp.solvers.keys(), case_sensitive=False),
    multiple=False,
    default=None,
)
@click.option(
    "-f",
    "--flavor",
    help="Higher-level strategy",
    type=click.Choice(usvp.flavors.keys(), case_sensitive=False),
    default="plain",
)
@click.option("-p", "--parallelism", help="Number of threads", default=1)
@click.option(
    "-P",
    "--params",
    help=(
        "Key-value pairs passed to the solver;"
        " values are `eval`d;"
        " pass multiple times for passing multiple parameters"
    ),
    type=(str, str),
    multiple=True,
    default=None,
)
@click.option(
    "--loglvl",
    help="Level of verbosity",
    type=click.Choice(["DEBUG", "INFO", "WARNING"], case_sensitive=False),
    default="INFO",
)
@click.argument("filename", type=str)
def solve(curve, m, algorithm, flavor, parallelism, params, loglvl, filename):
    """Solve instance provided as a text file."""

    logging.basicConfig(level=loglvl, format="%(message)s")
    install_mp_handler()

    from ecdsa_hnp import ECDSA, ECDSASolver

    ecdsa = ECDSA(curve=curve)
    with open(filename, "r") as f:
        lines = f.readlines()
        if m is None:
            m = len(lines)
        solver = ECDSASolver(ecdsa, lines=lines, m=m, threads=parallelism)
        if params is None:
            params = ""
        if params:
            params = dict([(x, eval(y)) for x, y in params])
        else:
            params = {}
        key, res = solver(solver=algorithm, flavor=flavor, **params)
        if res.success:
            print("Success. Secret key:", hex(key))
        else:
            print("Failed.")


@ecdsa.command(context_settings=dict(ignore_unknown_options=True))
@click.option("-n", "--nlen", help="Bit length of curve", default=256)
@click.option("-k", "--klen", help="Number of nonce bits", default=128, type=float)
@click.option("-m", "--m", default=2, help="Number of samples considered per instance")
@click.option("-s", "--skip", multiple=True, type=str, help="skip estimating this solver")
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
@click.option(
    "--loglvl",
    help="Level of verbosity",
    type=click.Choice(["DEBUG", "INFO", "WARNING"], case_sensitive=False),
    default="INFO",
)
def estimate(nlen, klen, m, skip, loglvl, args):
    """
    Estimate the cost.

    Fractional nonce lengths (`-k`) are interpreted as a fraction of larger nonces.  For example
    `252.2` means `0.2⋅m` nonces of size `2^{253}` and `0.8⋅m` of size `2^{252}`.

    """
    logging.basicConfig(level=loglvl)
    from ecdsa_hnp import estimate

    if klen >= nlen:
        raise ValueError("{klen:.2f} ≥ {nlen}".format(klen=klen, nlen=nlen))
    if args:
        logging.warning("% warning: ignoring {args}".format(args=args))
    return estimate(nlen=nlen, m=m, klen=klen, skip=skip)


@ecdsa.command()  # noqa: C901
@click.argument("filename", type=str)
@click.option(
    "-p",
    "--predicate",
    help="Predicate to decide if a statistic should be considered",
    type=str,
    default="bool(stat)",
)
@click.option(
    "-c",
    "--csvfilename",
    help="If a filename is passed, a csv file is written with the data gathered from parsing",
    type=str,
    default=None,
)
def parse_benchmark(filename, predicate, csvfilename):
    """
    Parse output of `benchmark` command.
    """
    stats = OrderedDict()

    @total_ordering
    @dataclass
    class Stats:
        date: str
        host: str
        nlen: int
        m: int
        errors: float
        klen: float
        alg: str
        seed: int
        params: str
        tag: int
        complete: bool = False
        successes: Accumulator = Accumulator(0, repr="sum", count=False)
        trials: Accumulator = Accumulator(0, repr="sum", count=False)
        cputime = Accumulator(0, repr="avg", count=False)
        walltime = Accumulator(0, repr="avg", count=False)
        work = Accumulator(0, repr="avg", count=False)
        v_over_b0 = Accumulator(0, repr="avg", count=False)

        def __repr__(self):
            if float(self.trials) == 0:
                return (
                    "{{"
                    "tag: 0x{stat.tag:016x}, "
                    "nlen: {stat.nlen:3d}, klen: {stat.klen:.3f}, m: {stat.m:3d}, "
                    "  NO DATA   "
                    'alg: "{stat.alg}", params: {{{stat.params}}}'
                    "}}".format(stat=self)
                )

            return (
                "{{"
                "tag: 0x{stat.tag:016x}, "
                "nlen: {stat.nlen:3d}, klen: {stat.klen:.3f}, m: {stat.m:3d}, "
                "e: {stat.errors:.3f}, "
                "successes: {stat.successes.sum:4.0f}, sr: {sr:5.1f}%, "
                "work: {work:>6s}, sf: {sf:.2f}, "
                "ct: {ct:10.2f}s, ct/sr: {ctsr:10.2f}s, "
                "wt: {wt:10.2f}s, wt/sr: {wtsr:10.2f}s, "
                'alg: "{stat.alg}", params: {{{stat.params}}}'
                "}}".format(
                    stat=self,
                    sr=self.sr * 100,
                    work="2^%.1f" % log(self.work.avg, 2),
                    sf=self.v_over_b0.avg if self.v_over_b0._ctr else 0.0,
                    ct=self.ct("s"),
                    ctsr=self.ctsr("s"),
                    wt=self.wt("s"),
                    wtsr=self.wtsr("s"),
                )
            )

        def __bool__(self):
            return float(self.trials) != 0.0

        def __lt__(self, other):
            return (self.nlen, self.klen, self.m, self.alg) < (
                other.nlen,
                other.klen,
                other.m,
                self.alg,
            )

        def ct(self, unit="m"):
            if unit == "s":
                return float(self.cputime)
            if unit == "m":
                return ceil(float(self.cputime) / 60)
            elif unit == "h":
                return ceil(float(self.cputime) / 3600)
            elif unit == "d":
                return ceil(float(self.cputime) / 24 / 3600)
            else:
                raise ValueError(unit)

        def wt(self, unit="m"):
            if unit == "s":
                return float(self.walltime)
            if unit == "m":
                return ceil(float(self.walltime) / 60)
            elif unit == "h":
                return ceil(float(self.walltime) / 3600)
            elif unit == "d":
                return ceil(float(self.walltime) / 24 / 3600)
            else:
                raise ValueError(unit)

        def ctsr(self, unit="m"):
            return ceil(self.ct(unit=unit) / self.sr) if self.sr != 0.0 else 0

        def wtsr(self, unit="m"):
            return ceil(self.wt(unit=unit) / self.sr) if self.sr != 0.0 else 0

        @property
        def sr(self):
            if float(self.trials) == 0:
                return 0.0
            else:
                return float(self.successes) / float(self.trials)

    def parse_stat(line):
        pattern = (
            "% ([^ ]+ [^ ]+) ([^ ]+) :: "
            "nlen: ([0-9]+), m:\\s+([0-9]+), "
            "klen: ([0-9\\.]+), alg: ([^ ]+), "
            "seed: ([0-9a-fA-Fx]+), "
            "params: {(.*?)}"
        )

        pattern_w_tag = (
            "% ([^ ]+ [^ ]+) ([^ ]+) ([^ ]+) :: "
            "nlen: ([0-9]+), m:\\s+([0-9]+), "
            "klen: ([0-9\\.]+), e: ([0-9\\.]+), alg: ([^ ]+), "
            "seed: ([0-9a-fA-Fx]+), "
            "params: {(.*?)}"
        )

        if re.match(pattern, line):
            date, host, nlen, m, klen, alg, seed, params = re.match(pattern, line).groups()
            tag = abs(hash((date, host, nlen, m, klen, alg, seed, params)))
        elif re.match(pattern_w_tag, line):
            date, host, tag, nlen, m, klen, errors, alg, seed, params = re.match(
                pattern_w_tag, line
            ).groups()
            tag = int(tag, 16)

        if params:
            params_ = []
            for param in params.split(","):
                k, v = param.strip().split(":")
                k = k.strip()[1:-1]
                if "'" not in v:
                    v = "%.2f" % float(v)
                else:
                    v = v.strip()[1:-1]
                params_.append((k, v))
            params = ", ".join(["{k}: {v:}".format(k=k, v=v) for k, v in params_])

        return Stats(
            date=date,
            host=host,
            nlen=int(nlen),
            m=int(m),
            klen=float(klen),
            errors=float(errors),
            alg=alg,
            seed=int(seed, 16),
            params=params,
            tag=tag,
        )

    def parse_experiment(stat, line):
        pattern = (
            "try: .*,(?: tag: .*,|) success: ([01]), .*, \\|v\\|/\\|b\\[0\\]\\|: ([0-9\\.]+)"
            ".*, cpu: +([0-9\\.]+)s, wall: +([0-9\\.]+)s, work: +([0-9]+)"
        )
        success, v_over_b0, cpu, wall, work = re.match(pattern, line).groups()
        stat.trials += 1
        stat.successes += int(success)
        # if int(success):
        stat.v_over_b0 += float(v_over_b0)
        stat.cputime += float(cpu)
        stat.walltime += float(wall)
        stat.work += int(work)
        return stat

    with open(filename, "r") as fh:
        for line in fh.readlines():
            if line.startswith("%") and "seed" in line:
                stat = parse_stat(line)
                stats[stat.tag] = stat
                stats["prev"] = stat

            if line.startswith("try:"):
                if "tag" in line:
                    tag = re.match(".*tag: ([^,]+).*", line).groups()[0]
                    stat = stats[int(tag, 16)]
                    stat = parse_experiment(stat, line)
                else:
                    stat = parse_experiment(stats["prev"], line)

            if line.startswith("%") and "sr" in line:
                matches = re.match(".* (0x.*?) ::.*", line)
                if matches:
                    tag = int(matches.groups()[0], 16)
                    stats[tag].complete = True

    for stat in sorted(stats.values()):
        if stat.tag != "prev" and eval(predicate):
            print(stat)

    if csvfilename:
        with open(csvfilename, "w") as fh:
            writer = csv.writer(fh, delimiter=",")
            writer.writerow(
                [
                    "nlen",
                    "klen",
                    "m",
                    "e",
                    "alg",
                    "sr",
                    "work",
                    "ct",
                    "wt",
                    "ctsr",
                    "wtsr",
                    "params",
                ]
            )

            for stat in sorted(stats.values()):
                if stat.tag != "prev" and eval(predicate):
                    writer.writerow(
                        [
                            stat.nlen,
                            stat.klen,
                            stat.m,
                            stat.errors,
                            stat.alg,
                            "%.2f" % stat.sr,
                            "%.2f" % stat.work,
                            "%.2f" % stat.ct("s"),
                            "%.2f" % stat.ctsr("s"),
                            "%.2f" % stat.wt("s"),
                            "%.2f" % stat.wtsr("s"),
                            "'%s'" % stat.params,
                        ]
                    )


if __name__ == "__main__":
    ecdsa()
