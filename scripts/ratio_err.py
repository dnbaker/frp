import sys
import numpy as np
from subprocess import check_call

def load_data(path):
    with open(path) as f:
        return np.array([list(map(float, i[:-1].split(", "))) for i in f if i[0] != "#" and "1e-300" not in i], dtype=np.double)
    

def print_results(path, sigma, n, ofp):
    print("loading data")
    data = load_data(path)
    print("loaded data")
    mean_fac = np.mean(data[:,1] / data[:,2])
    ofp.write("%f\t%f\t%i\n" % (mean_fac, sigma, n))
    print("mf: %f. %f, %f\n" % (mean_fac,  np.mean(data[:,1]), np.mean(data[:,2])))
    ratios = data[:,0]
    print("ratmean %f, ratstd %f" % (np.mean(ratios), np.std(ratios)))
    #data[2,:] /= np.mean(data[2,:])
    #data[1,:] /= np.mean(data[1,:])
    print(data)
    print("n: %i. corr: %e. ratio off %e. size %e" % (
        len(data), np.corrcoef(data[:,1], data[:,2])[0,1], mean_fac, len(data[:,1])))
    inv =  np.reciprocal(data[:,2])
    inv[inv > 1e30] = 1e-300
    print(inv)
    print("inv n: %i. corr: %e. ratio off %e. size %e" % (
        len(data), np.corrcoef(data[:,1], inv)[0,1], mean_fac, len(data[:,1])))
    tmp = data[:,1]
    tmp[tmp == 0] = 1e-300
    log = np.log(tmp)
    print("log'd n: %i. corr: %e. ratio off %e. size %e" % (
        len(data), np.corrcoef(log, data[:,2])[0,1], mean_fac, len(data[:,1])))
    log = -log
    print("-log'd n: %i. corr: %e. ratio off %e. size %e" % (
        len(data), np.corrcoef(log, data[:,2])[0,1], mean_fac, len(data[:,1])))
    print(log)

def submit_work(tup):
    sig, SIZE, fn = tup[:]
    s  = "kernel_test -i128 -s %f -S %i  > %s" % (sig, SIZE, fn)
    print("Calling %s!" % s)
    check_call(s, shell=True)
    return fn

def main():
    import multiprocessing
    SIGS = [1, 10, 100, 10000]
    SIZE = 1 << 16
    ratsigf = open("ratsig.%s.txt" % (SIZE), "w")
    ratsigf.write("#Ratio\tSigma\tN\n")
    spool = multiprocessing.Pool(8)
    fns = spool.map(submit_work,
                    [[sig, SIZE, "output.%s.txt" % (sig)] for sig in SIGS])
    [print_results(fn, sig, SIZE, ratsigf) for sig, fn in zip(SIGS, fns)]

if __name__ == "__main__":
    main()
