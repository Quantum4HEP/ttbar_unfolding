import ROOT
import numpy as np
from qunfold.root2numpy import TH2_to_numpy, TH1_to_numpy, TVector_to_numpy
from qunfold.utils import normalize_response, compute_chi2
from qunfold import QPlotter
from utils import *
import pylab as plt
from unfolding_functions import roounfolder, qunfolder

##################CONFIGS#####################
# Pick a variable (ex. c_thetap or ttbar_mass)
variable = "ttbar_mass"

y_axis_min=0
y_axis_max=0.14
#######################################


# Load RooUnfold library
if ROOT.gSystem.Load("libRooUnfold"):
    raise ImportError("RooUnfold was not loaded successfully!")

# Open input ROOT file
root_file = ROOT.TFile.Open("ttbar_qunfold.root")

# Get binning for histograms
overflow = True
binning = get_binning(root_file=root_file, variable=variable, overflow=overflow)

# Read particle-level(=measured) histograms
dir = "particle"
th1_measured = root_file.Get(f"{dir}/{variable}")
th1_fake = root_file.Get(f"{dir}/{variable}_fake")
d = TH1_to_numpy(th1_measured, overflow=overflow)
fake = TH1_to_numpy(th1_fake, overflow=overflow)

# Read parton-level(=truth) histograms
dir = "parton"
th1_truth = root_file.Get(f"{dir}/{variable}_parton")
th1_miss = root_file.Get(f"{dir}/{variable}_miss")
t = TH1_to_numpy(th1_truth, overflow=overflow)
miss = TH1_to_numpy(th1_miss, overflow=overflow)

# Read Monte Carlo migration matrix (measured X truth)
dir = "migration"
th2_migration = root_file.Get(f"{dir}/{variable}_migration")
M = TH2_to_numpy(th2_migration, overflow=overflow)

# Get Monte Carlo reco and truth histograms
th1_proj_reco_mc = th2_migration.ProjectionX()
th1_reco_mc = th1_proj_reco_mc.Clone("th1_reco_mc")
th1_reco_mc.Add(th1_fake)
th1_proj_truth_mc = th2_migration.ProjectionY()
th1_truth_mc = th1_proj_truth_mc.Clone("th1_truth_mc")
th1_truth_mc.Add(th1_miss)
mu = TH1_to_numpy(th1_reco_mc, overflow=overflow)
x = TH1_to_numpy(th1_truth_mc, overflow=overflow)

# Normalize migration matrix in numpy
proj_truth_mc = TH1_to_numpy(th1_proj_truth_mc)
M = normalize_response(response=M, truth_mc=proj_truth_mc)

# Create response matrix object in RooUnfold
roo_response = ROOT.RooUnfoldResponse(th1_reco_mc, th1_truth_mc, th2_migration)

# Estimate purity from Monte Carlo data
purity = estimate_purity(reco=mu, fake=fake)
roo_purity = TVector_to_numpy(roo_response.Vpurity())

# Estimate efficiency from Monte Carlo data
efficiency = estimate_efficiency(truth=x, miss=miss)
roo_efficiency = TVector_to_numpy(roo_response.Vefficiency())

# Sanity checks
assert np.allclose(np.sum(M, axis=0), 1)
# assert np.allclose(purity[1:-1], roo_purity)
# assert np.allclose(efficiency[1:-1], roo_efficiency)
assert np.allclose(mu, M @ (x - miss) + fake)
assert np.allclose(mu, M @ (x * efficiency) / purity)

# Run Matrix Inversion unfolding in numpy
#np_unfolded = (1 / efficiency) * (np.linalg.inv(M) @ (d * purity))

#Passing from migration to response and correcting fake events
resp = M * efficiency
meas = d - fake

#run RooUnfold and QUnfold algorithm
sol_mi, cov_mi = roounfolder("MI", roo_response, th1_measured, th1_fake)
sol_ibu, cov_ibu = roounfolder("IBU", roo_response, th1_measured, th1_fake)
sol_grb, cov_grb = qunfolder("GRB", resp, d, fake, binning)

#calculating error and chi square
err_mi = np.sqrt(np.diag(cov_mi))
chi2_mi = compute_chi2(observed=sol_mi, expected=t)
err_ibu= np.sqrt(np.diag(cov_ibu))
chi2_ibu = compute_chi2(observed=sol_ibu, expected=t)
err_grb= np.sqrt(np.diag(cov_grb))[1:-1]
chi2_grb = compute_chi2(observed=sol_grb[1:-1], expected=t[1:-1])

#Plot just one method and save the response
qplotter = QPlotter(
    response=M,
    measured=d,
    truth=t,
    unfolded=sol_grb,
    covariance=cov_grb,
    binning=binning,
    method="GRB",
    norm=True,
)
qplotter.save_response(f"{variable}_response.pdf")
#qplotter.show_histograms()


#plotting and comparing the results
fig = plt.figure(figsize=(12, 9))
gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[3, 1], hspace=0)
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1], sharex=ax1)

QPlotter.histogram_plot(ax=ax1, xedges=binning[1:-1], hist=t[1:-1], label="Truth", norm=True)
QPlotter.histogram_plot(ax=ax1, xedges=binning[1:-1], hist=d[1:-1], label="Measured", norm=True)

binning = binning[1:-1]
xlims = (binning[1], binning[-2])
widths = np.diff(binning)
xmid = binning[:-1] + 0.5 * widths
ax1.set_ybound(y_axis_min,y_axis_max)


#Add a QPlotter.errorbar_plot for each unfolding method
QPlotter.errorbar_plot(ax=ax1, xmid=xmid, hist=sol_grb[1:-1], err=err_grb, xlims=xlims, label="GRB", chi2=chi2_grb, norm=True)
QPlotter.errorbar_plot(ax=ax1, xmid=xmid, hist=sol_mi[1:-1], err=err_mi, xlims=xlims, label="MI", chi2=chi2_mi, norm=True)
QPlotter.errorbar_plot(ax=ax1, xmid=xmid, hist=sol_ibu[1:-1], err=err_ibu, xlims=xlims, label="IBU", chi2=chi2_ibu, norm=True)


#Add a ratio plot
sol_ratio = sol_grb[1:-1] / t[1:-1]
err_ratio = err_grb / t[1:-1]
QPlotter.ratio_plot(ax=ax2, xmid=xmid, ratio=sol_ratio, err=err_ratio, label="GRB", xticks=binning)

fig.savefig(f"{variable}_unfolding.pdf")







