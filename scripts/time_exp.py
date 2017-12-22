#/usr/bin/env python
import sys
import numpy as np
import subprocess

class settings:
    def __init__(self, sigma, insize, outsize):
        self.sigma = sigma
        self.insize = insize
        self.outsize = outsize
    def __str__(self):
        return "%f|%i|%i" % (self.sigma, self.insize, self.outsize)


def get_output(sigma, insize, outsize):
    s = "kernel_time -Oi%i -s %f -S %i" % (insize, sigma, outsize)
    output_str = [i.strip() for i in subprocess.check_output(s,
                  shell=True).decode().split("\n") if i]
    return output_str, settings(sigma, insize, outsize)


def submit_work(tup):
    return get_output(*tup)


def main():
    SIGS = [0.25, 0.5, 1., 2., 5.]
    INSIZES = [128, 256, 1024, 4096]
    START = 4096
    OUTSIZES = [START << 1, START << 2, START << 4, START << 6]
    '''
    SIGS = [0.5]
    INSIZES = [64, 128]
    OUTSIZES = [1024, 2048]
    '''
    subgen = [(sig, ins, outs) for sig in SIGS for ins in INSIZES for outs in OUTSIZES if ins * outs < 4096 * 131072]
    import multiprocessing
    spool = multiprocessing.Pool(8)
    strings = spool.map(submit_work, subgen)
    main_dict = {}
    for string, setting in strings:
        for key, val in zip(string[0].split(), string[1].split()):
            main_dict[
                str(setting)] = {
                    k: float(v) for
                    k, v in zip(string[0].split(), string[1].split())}
    print("Input Size\tOutput Size\tSigma\tRFF Time\tORF Time"
          "\tSORF Time\tFF Time\tFastest method\t"
          "Ratio of fastest/slowest\tRatio of SORF over slowest")
    keys = ["rf", "orf", "sorf", "ff"]
    for sig, i, o in subgen:
        subdict = main_dict[str(settings(sig, i, o))]
        entries = [subdict[k] for k in keys]
        besti = np.argmin(entries)
        ratio = np.max(entries) / entries[besti]
        linestr = "%i\t%i\t%f" % (i, o, sig)
        entrystr = "\t".join(list(map(str, entries)))
        endstr = "%s\t%f\t%f" % (keys[besti], ratio, np.max(entries) / subdict["sorf"])
        print("\t".join([linestr, entrystr, endstr]))


if __name__ == "__main__":
    main()
