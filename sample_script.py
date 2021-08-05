#!/usr/bin/env python3
from ecdsa_hnp import ECDSA, ECDSASolver

ecdsa = ECDSA(nbits=256)
lines, k_list, d = ecdsa.sample(m=70, is_msb=False)
solver = ECDSASolver(ecdsa, lines, m=70, is_msb=False)
key, res = solver("bkz-enum")
if res.success:
    assert (key == d)
    print("Found key:", key)
else:
    print("Failed")
