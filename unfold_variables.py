import ROOT
import numpy as np
from qunfold import QUnfolder, QPlotter
from qunfold.root2numpy import *
from utils import get_binning


# Open input ROOT file
root_file = ROOT.TFile.Open("ttbar_qunfold.root")

# Consider left-most and right-most overflow bins
overflow = True

# Loop over variables to be unfolded
for variable in ["c_thetap", "ttbar_mass"]:

    # Get binning from ROOT histograms
    binning = get_binning(root_file, variable=variable, overflow=overflow)

    # Get all required data as ROOT histograms
    th2_migration = root_file.Get(f"migration/{variable}_migration")
    th1_measured = root_file.Get(f"particle/{variable}")
    th1_fake = root_file.Get(f"particle/{variable}_fake")
    th1_truth = root_file.Get(f"parton/{variable}_parton")
    th1_miss = root_file.Get(f"parton/{variable}_miss")

    # Apply corrections to consider miss and fake events
    th1_truth = th1_truth - th1_miss
    th1_measured = th1_measured - th1_fake

    # Convert ROOT histograms to numpy arrays
    response = TH2_to_numpy(th2_migration, overflow=overflow)
    measured = TH1_to_numpy(th1_measured, overflow=overflow)
    truth = TH1_to_numpy(th1_truth, overflow=overflow)

    # Normalize the response(=migration) matrix
    norms = np.sum(response, axis=0)
    norms[norms == 0] = 1e-10
    response = response / norms

    # Create the RooUnfoldResponse object
    empty = th1_measured.Clone("h")
    empty.Reset()
    roo_response = ROOT.RooUnfoldResponse(empty, empty, th2_migration)
    roo_response.UseOverflow(overflow)

    # Run Matrix Inversion (MI) unfolding
    unfolder = ROOT.RooUnfoldInvert(roo_response, th1_measured)
    roo_unfolded = unfolder.Hunfold()
    roo_covariance = unfolder.Eunfold()
    unfolded = TH1_to_numpy(roo_unfolded, overflow=overflow)
    covariance = TMatrix_to_numpy(roo_covariance)
    qplotter = QPlotter(
        response=response,
        measured=measured,
        truth=truth,
        unfolded=unfolded,
        covariance=covariance,
        binning=binning,
        method="MI",
        norm=False
    )
    qplotter.show_histograms()

    # Run Iterative Bayesian Unfolding (IBU)
    unfolder = ROOT.RooUnfoldBayes(roo_response, th1_measured)
    unfolder.SetIterations(4)
    unfolder.SetVerbose(0)
    roo_unfolded = unfolder.Hunfold()
    roo_covariance = unfolder.Eunfold()
    unfolded = TH1_to_numpy(roo_unfolded, overflow=overflow)
    covariance = TMatrix_to_numpy(roo_covariance)
    qplotter = QPlotter(
        response=response,
        measured=measured,
        truth=truth,
        unfolded=unfolded,
        covariance=covariance,
        binning=binning,
        method="IBU",
        norm=False
    )
    qplotter.show_histograms()

    # Run Singular Values Decomposition (SVD) unfolding
    unfolder = ROOT.RooUnfoldSvd(roo_response, th1_measured)
    unfolder.SetKterm(2)
    unfolder.SetVerbose(0)
    roo_unfolded = unfolder.Hunfold()
    roo_covariance = unfolder.Eunfold()
    unfolded = TH1_to_numpy(roo_unfolded, overflow=overflow)
    covariance = TMatrix_to_numpy(roo_covariance)
    qplotter = QPlotter(
        response=response,
        measured=measured,
        truth=truth,
        unfolded=unfolded,
        covariance=covariance,
        binning=binning,
        method="SVD",
        norm=False
    )
    qplotter.show_histograms()

    # Run the QUnfold algorithm using Gurobi solver
    qunfolder = QUnfolder(
        response=response, measured=measured, binning=binning, lam=0.0)
    unfolded, covariance = qunfolder.solve_gurobi_integer()
    qplotter = QPlotter(
        response=response,
        measured=measured,
        truth=truth,
        unfolded=unfolded,
        covariance=covariance,
        binning=binning,
        method="GRB",
        norm=False
    )
    qplotter.show_histograms()

    # Run the QUnfold algorithm using D-Wave hybrid sampler
    # D-Wave account configuration using valid API token is required
    # See https://docs.ocean.dwavesys.com/en/stable/overview/sapi.html#create-a-configuration-file
    qunfolder = QUnfolder(
        response=response, measured=measured, binning=binning, lam=0.0)
    unfolded, covariance = qunfolder.solve_hybrid_sampler()
    qplotter = QPlotter(
        response=response,
        measured=measured,
        truth=truth,
        unfolded=unfolded,
        covariance=covariance,
        binning=binning,
        method="HYB",
        norm=False
    )
    qplotter.show_histograms()
