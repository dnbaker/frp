import sys
import numpy as np

def load_data(path):
    with open(path) as f:
        return np.array([list(map(float, i[:-1].split(", "))) for i in f if i[0] != "#"], dtype=np.double)
    

if __name__ == "__main__":
    argc, argv = len(sys.argv), sys.argv
    data = load_data(argv[1])
    mean_fac = np.mean(data[:,1]) / np.mean(data[:,2])
    print("mf: %f. %f, %f\n", mean_fac,  np.mean(data[:,1]), np.mean(data[:,2]))
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
    print("n: %i. corr: %e. ratio off %e. size %e" % (
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
