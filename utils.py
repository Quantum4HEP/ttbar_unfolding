import ROOT
import awkward
from array import array
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
    kbd.fit(data.reshape(-1, 1))
    return kbd.bin_edges_[0]


def create_histo(name, title, binning, data, mask):
    nbins = len(binning) - 1
    bins = array("f", binning)
    histo = ROOT.TH1F(name, title, nbins, bins)
    for x in data[mask]:
        histo.Fill(x)
    return histo


def create_response(name, title, binning, data_reco, data_truth, mask):
    nbins = len(binning) - 1
    bins = array("f", binning)
    response = ROOT.TH2F(name, title, nbins, bins, nbins, bins)
    for xr, xt in zip(data_reco[mask], data_truth[mask]):
        response.Fill(xr, xt)
    return response
