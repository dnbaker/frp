
def generate_code(j, k, s1, array):
    jk = j + k
    jks = j + k + s1
    return "    u = {array}[x + {jk}], v = {array}[x + {jks}];\n".format(**locals()) + \
        "    {array}[x + {jk}] = u + v;\n".format(**locals()) + \
        "    {array}[x + {jks}] = u - v;\n".format(**locals())


def make_one_levelset(level, neleml2, array="array"):
    n = 1 << neleml2
    s1 = 1 << level
    s2 = s1 << 1
    assert n >= s2, "n: {n}. s2: {s2}".format(**locals())
    ret = ""
    level_offsets = range(0, n, s2)
    sublevel_offsets = range(s1)
    for lo in level_offsets:
        for so in sublevel_offsets:
            ret += generate_code(lo, so, s1, array)
    return ret

def make_unrolled_fht(level: int, neleml2: int, array="array"):
    return "    T u, v;\n" + "".join(make_one_levelset(l, neleml2, array) for l in range(level + 1))


if __name__ == "__main__":
    import sys
    nk = int(sys.argv[1]) if sys.argv[1:] else 4
    kernels = (make_one_levelset(i, i + 1) for i in range(nk))
    for ind, k in enumerate(kernels):
        with open("fht_kernel%d.cu" % ind, "w") as f:
            f.write(k)
