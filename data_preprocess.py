import ROOT
import uproot
from utils import *

######################################## CONFIGS ########################################
input_rootfile_path = "./entangled_ttbar_atlas.root"
output_rootfile_path = "./ttbar_qunfold.root"

variables = [
    Variable(name="c_thetap", nbins=20, bounds=(-1, 1)),
    Variable(name="ttbar_mass", nbins=20, bounds=(300, 2000)),
]

# Useful to cut events over a certain energy level
en_min = 0
#########################################################################################


# Remove statistics info box from ROOT histograms
ROOT.gStyle.SetOptStat(0)

# Open input ROOT file and get tree
tree = uproot.open(input_rootfile_path)["particle_level"]

# Open new output ROOT file and make directories
output_rootfile = ROOT.TFile(output_rootfile_path, "RECREATE")
particle_dir = output_rootfile.mkdir("particle")
parton_dir = output_rootfile.mkdir("parton")
migration_dir = output_rootfile.mkdir("migration")

# Get boolean mask to identify particle and/or parton events
particle = get_numpy_array(tree=tree, varname="passed_particle_sel", dtype=bool)
parton = get_numpy_array(tree=tree, varname="passed_parton_sel", dtype=bool)


# Get events weight (negative for background process events)
event_weight = get_numpy_array(tree=tree, varname="eventweight")

for var in variables:
    # Get numpy arrays for particle and parton events
    arr_particle = get_numpy_array(tree=tree, varname=var.name)
    arr_parton = get_numpy_array(tree=tree, varname=var.name_parton)
    arr_mass = get_numpy_array(tree=tree, varname="ttbar_mass_parton")
    

    # Set proper binning according to data distribution
    binning = find_binning(x=arr_parton, nbins=var.nbins, bounds=var.bounds)

    # Create and fill histograms for particle(=measured) and parton(=truth) events
    th1_particle = create_histo(
        name=var.name,
        title=var.name,
        binning=binning,
        data=arr_particle,
        weights=event_weight,
        mask=particle #& (arr_mass>en_min),
    )
    th1_parton = create_histo(
        name=var.name_parton,
        title=var.name_parton,
        binning=binning,
        data=arr_parton,
        weights=event_weight,
        mask=parton #& (arr_mass>en_min),
    )

    # Create and fill missed and fake events histograms
    th1_miss = create_histo(
        name=var.name + "_miss",
        title=var.name + "_miss",
        binning=binning,
        data=arr_parton,
        weights=event_weight,
        mask=~particle & parton #& (arr_mass>en_min),
    )
    th1_fake = create_histo(
        name=var.name + "_fake",
        title=var.name + "_fake",
        binning=binning,
        data=arr_particle,
        weights=event_weight,
        mask=particle & ~parton# & (arr_mass>en_min),
    )

    # Create and fill migration matrix histogram (measured X truth)
    migration = create_migration(
        name=var.name + "_migration",
        title=var.name + "_migration",
        binning=binning,
        data_reco=arr_particle,
        data_truth=arr_parton,
        weights=event_weight,
        mask=particle & parton# & (arr_mass>en_min),
    )

    # Write histograms to output ROOT file
    particle_dir.cd()
    th1_particle.Write()
    th1_fake.Write()

    parton_dir.cd()
    th1_parton.Write()
    th1_miss.Write()

    migration_dir.cd()
    migration.Write()

output_rootfile.Close()
