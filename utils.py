import ROOT
import awkward
from array import array
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer


class Variable:
    def __init__(self, name, nbins, bounds):
        self.name = name
        self.name_parton = name + "_parton"
        self.nbins = nbins
        self.bounds = bounds


def get_numpy_array(tree, varname, dtype=float):
    awkarr = tree[varname].array()
    return awkward.to_numpy(awkarr).astype(dtype)


def find_binning(x, nbins, bounds):
    kbd = KBinsDiscretizer(nbins, encode="ordinal", strategy="kmeans")
    low, high = bounds
    data = x[(low <= x) & (x <= high)]
    bin_egdes = kbd.fit(data.reshape(-1, 1)).bin_edges_[0]
    bin_egdes = [low] + bin_egdes.tolist()[1:-1] + [high]
    return bin_egdes


def create_histo(name, title, binning, data, weights, mask):
    nbins = len(binning) - 1
    bins = array("f", binning)
    histo = ROOT.TH1F(name, title, nbins, bins)
    for x, w in zip(data[mask], weights[mask]):
        histo.Fill(x, w)
    return histo


def create_migration(name, title, binning, data_reco, data_truth, weights, mask):
    nbins = len(binning) - 1
    bins = array("f", binning)
    migration = ROOT.TH2F(name, title, nbins, bins, nbins, bins)
    for xr, xt, w in zip(data_reco[mask], data_truth[mask], weights[mask]):
        migration.Fill(xr, xt, w)
    return migration


def get_binning(th1_histo, overflow=True):
    xaxis = th1_histo.GetXaxis()
    nbins = xaxis.GetNbins()
    bin_edges = [xaxis.GetBinLowEdge(i) for i in range(1, nbins + 1)]
    bin_edges += [xaxis.GetBinUpEdge(nbins)]
    if overflow:
        bin_edges = [-np.inf] + bin_edges + [np.inf]
    return np.array(bin_edges)


def estimate_efficiency(reco, miss):
    eff = np.ones_like(reco)
    num = reco
    den = reco + miss
    nonzero = np.nonzero(den)
    eff[nonzero] = num[nonzero] / den[nonzero]
    return eff


def estimate_purity(reco, fake):
    # maybe wrong, work in progress...
    pur = np.ones_like(reco)
    num = reco
    den = reco + fake
    nonzero = np.nonzero(den)
    pur[nonzero] = num[nonzero] / den[nonzero]
    return pur
