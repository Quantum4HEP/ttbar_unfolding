import ROOT
import uproot
from utils import *

######################################## CONFIGS ########################################
input_rootfile_path = "./entangled_ttbar_atlas.root"
output_rootfile_path = "./ttbar_qunfold.root"

variables = [
    Variable(name="c_thetap", nbins=15, bounds=(-1, 1)),
    Variable(name="ttbar_mass", nbins=20, bounds=(300, 2000)),
]
#########################################################################################


# Remove statistics info box from ROOT histograms
ROOT.gStyle.SetOptStat(0)

# Open input ROOT file and get tree
tree = uproot.open(input_rootfile_path)["particle_level"]

# Open new output ROOT file and make directories
output_rootfile = ROOT.TFile(output_rootfile_path, "RECREATE")
particle_dir = output_rootfile.mkdir("particle")
parton_dir = output_rootfile.mkdir("parton")

# Get boolean mask to identify particle and/or parton events
particle = get_numpy_array(tree=tree, varname="passed_particle_sel", dtype=bool)
parton = get_numpy_array(tree=tree, varname="passed_parton_sel", dtype=bool)

# Get boolean mask to identify background events
background = get_numpy_array(tree=tree, varname="eventweight") < 0

for var in variables:
    # Get numpy arrays for particle and parton events
    arr_particle = get_numpy_array(tree=tree, varname=var.name)
    arr_parton = get_numpy_array(tree=tree, varname=var.name_parton)

    # Set proper binning according to data distribution
    binning = find_binning(x=arr_parton, nbins=var.nbins, bounds=var.bounds)

    # Create and fill histograms for particle(=measured) and parton(=truth) events
    th1_particle = create_histo(
        name=var.name,
        title=var.name,
        binning=binning,
        data=arr_particle,
        mask=particle,
    )
    th1_parton = create_histo(
        name=var.name_parton,
        title=var.name_parton,
        binning=binning,
        data=arr_parton,
        mask=parton,
    )

    # Create and fill histograms for particle and parton background events
    th1_bkg_particle = create_histo(
        name=var.name + "_bkg",
        title=var.name + "_bkg",
        binning=binning,
        data=arr_particle,
        mask=background,
    )
    th1_bkg_parton = create_histo(
        name=var.name + "_bkg_parton",
        title=var.name + "_bkg_parton",
        binning=binning,
        data=arr_parton,
        mask=background,
    )

    # Create and fill histograms for miss and fake events
    th1_miss = create_histo(
        name=var.name + "_miss",
        title=var.name + "_miss",
        binning=binning,
        data=arr_parton,
        mask=~particle & parton,
    )
    th1_fake = create_histo(
        name=var.name + "_fake",
        title=var.name + "_fake",
        binning=binning,
        data=arr_particle,
        mask=particle & ~parton,
    )

    # Create and fill response TH2 histogram (measured X truth)
    response = create_response(
        name=var.name + "_response",
        title=var.name + "_response",
        binning=binning,
        data_reco=arr_particle,
        data_truth=arr_parton,
        mask=particle & parton,
    )

    # Write histograms to output ROOT file
    particle_dir.cd()
    th1_particle.Write()
    th1_bkg_particle.Write()
    th1_miss.Write()
    th1_fake.Write()
    response.Write()
    parton_dir.cd()
    th1_parton.Write()
    th1_bkg_parton.Write()

output_rootfile.Close()
