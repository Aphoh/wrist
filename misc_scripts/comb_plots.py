from math import comb, factorial


for r in range(1, 9):
    for n in range(1, 16):
        print(f"n:{n: >2} r:{r} {factorial(r) * comb(n + r - 1, r - 1): >10}")