#!/usr/bin/env python3
from ecdsa_hnp import ECDSA, ECDSASolver, make_klen_list

ecdsa = ECDSA(nbits=256)
lines, k_list, d = ecdsa.sample(m=70, is_msb=False, klen_list=make_klen_list(248,70))
solver = ECDSASolver(ecdsa, lines, m=70)
key, res = solver("bkz-enum")
if res.success:
    assert (key == d)
    print("Found key:", key)
else:
    print("Failed")
