# Bounded Distance Decoding with Predicate

This repository contains the Python/Sagemath source code for solving Bounded Distance Decoding augmented with a predicate characterising the target as introduced in:

> **On Bounded Distance Decoding with Predicate: Breaking the "Lattice Barrier" for the Hidden Number Problem**  
>
> *Martin R. Albrecht and Nadia Heninger*  
>
> Lattice-based algorithms in cryptanalysis often search for a target vector satisfying integer linear constraints as a shortest or closest vector in some lattice.  In this work, we observe that these formulations may discard non-linear information from the underlying application that can be used to distinguish the target vector even when it is far from being uniquely close or short.  
>
> We formalize lattice problems augmented with a predicate distinguishing a target vector and give algorithms for solving instances of these problems. We apply our techniques to lattice-based approaches for solving the Hidden Number Problem, a popular technique for recovering secret DSA or ECDSA keys in side-channel attacks, and demonstrate that our algorithms succeed in recovering the signing key for instances that were previously believed to be unsolvable using lattice approaches. We carried out extensive experiments using our estimation and solving framework, which we also make available with this work.

If you use this library in your work please cite

  Martin R. Albrecht and Nadia Heninger. *On Bounded Distance Decoding with Predicate: Breaking the "Lattice Barrier" for the Hidden Number Problem*

**TODO** Update reference

## ECDSA with Partially Known Nonces

The flagship application of this work is solving ECDSA with known nonce bits. The `ecdsa_cli.py` script provides a high level entry point.

1. To get estimates for the running times of the different algorithms for a set of parameters, the `estimate` functionality can be invoked as

    ``` shell
    sage -python ecdsa_cli.py estimate -n 256 -k 252 -m 66
    ```

2. To run the solver on a randomly generated problem instance with these parameters, use the `benchmark` function:

    ``` shell
    sage -python ecdsa_cli.py benchmark -n 256 -k 252 -m 65 --loglvl DEBUG
    ```

    If the algorithm is not specified, the script will automatically choose one for you, but you can also specify your chosen algorithm on the command line

    ``` shell
    sage -python ecdsa_cli.py benchmark -n 256 -k 252 -m 65 --algorithm "enum_pred" --loglvl DEBUG
    ```

3. To actually compute the secret key from input provided in a file, you can use the `solve` function. You need to specify the curve to use by name:

    ``` shell
    sage -python ecdsa_cli.py solve -c secp256k1 sample_input.txt
    ```

    Each line of the file is a space-separated list of the bit length of the nonce, the hex-encoded hash used in the ECDSA signature, the hex-encoded ECDSA signature as (r,s) concatenated together, and the hex-encoded public key.

For the moment, our scripts assume the most significant bits of the nonce are 0. If your use case involves known nonzero most significant bits, least significant bits, or another case, you can either transform your signatures and hash values accordingly, or modify our script to implement that case.

The following example uses the `scale` strategy to continue searching until the solution is found, which can deal with errors in the data, and will parallelize the algorithm in 8 threads:

``` shell
sage -python ecdsa_cli.py solve -c secp256k1 -f scale -p 8 sample_input.txt
```

If you wish to write your own script to use our functions as a library, here is a small custom Python script that shows how to invoke the relevant functions to compute the secret key for some randomly generated data:
``` python
from ecdsa_hnp import ECDSA, ECDSASolver,make_klen_list

if  __name__=='__main__':
    k = 252
    m = 70
    ecdsa = ECDSA(nbits=256)
    lines, k_list, d = ecdsa.sample(m,make_klen_list(k,m))
    solver = ECDSASolver(ecdsa,lines,m=m)
    key, res = solver("bkz-enum")
    if res.success:
        print(hex(key))
    else:
        print("Failed")
```

## Implemented Algorithms

Our algorithms solve the unique shortest vector problem augmented with a predicate. Using Kannan's embedding this enables to solve bounded distance decoding augmented with a predicate.

- **Enumeration with Predicate**: This algorithm performs lattice-point enumeration within a ball of radius *R*, for each point *v* found it checks whether the predicate *f(⋅)* holds, i.e. if *f(v) = 1*.

- **Sieving with Predicate**: This algorithm performs lattice sieving followed by a check for each point *v* of norm bounded by *((4/3)^(1/2) ⋅ gh(Λ)* whether the predicate *f(⋅)* holds, i.e. if *f(v) = 1*.

- **BKZ** with sieving or enumeration followed by a check for each point *v* in the output basis whether the predicate *f(⋅)* holds, i.e. if *f(v) = 1*.

## How to Install/Run

This framework builds on

- [FPLLL](https://github.com/fplll/fplll) and [FyLLL](https://github.com/fplll/fpylll) for datastructures and lattice-point enumeration, and
- [G6k](https://github.com/fplll/g6k) for lattice sieving.

### Using Conda/Manually

``` shell
conda create -n bddp python=3.7
conda activate bddp
conda install -c conda-forge sage

git clone https://github.com/fplll/fplll
cd fplll
autoreconf -i
./configure --prefix=$SAGE_LOCAL --disable-static
make install
cd ..

git clone https://github.com/fplll/fpylll
cd fpylll
pip install -r requirements.txt
pip install -r suggestions.txt
python setup.py build
python setup.py -q install
cd ..
    
git clone https://github.com/fplll/g6k
cd g6k
make
pip install -r requirements.txt
./rebuild.sh
python setup.py build
python setup.py -q install 
cd ..
```

### Using Docker

Running

``` shell
docker run -ti --rm -v `pwd`:/bdd-predicate -w /bdd-predicate martinralbrecht/bdd-predicate
```

from the root directory of this repository  will start SageMath with recent versions of FPLLL, FPyLLL and G6K installed. Our code is available under `/bdd-predicate`. Thus, e.g.

``` python
cd /bdd-predicate
load("usvp.py")
```

will load it.
