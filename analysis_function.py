import ROOT
import numpy as np
from qunfold.utils import compute_chi2, normalize_response
from qunfold.root2numpy import TH2_to_numpy, TH1_to_numpy
import matplotlib.pyplot as plt

#partendo da le due reco come TH1 di root, ottengo la purity come array numpy
def get_purity(h_reco_mc, h_reco):
    h_purity = h_reco.Clone("h_purity")
    h_purity.Reset()

    for i in range(1, h_reco.GetNbinsX() + 1):
        num = h_reco_mc.GetBinContent(i)
        denom = h_reco.GetBinContent(i)
        if denom != 0:
            purity = num / denom
            h_purity.SetBinContent(i, purity)
            #print(purity)
        else:
            h_purity.SetBinContent(i, 0)
            h_purity.SetBinError(i, 0)

    purity = TH1_to_numpy(h_purity, overflow=True)
    return purity


#partendo da le due reco come TH1 di root, ottengo l,efficiency come array numpy
def get_efficiency(h_truth_mc, h_truth):
    efficiency = []

    for i in range(0, h_truth.GetNbinsX() + 2):
        num = h_truth_mc.GetBinContent(i)
        denom = h_truth.GetBinContent(i)
        if denom != 0 :
            efficiency.append(num/denom)
        else:
            efficiency.append(1) 

    return np.array(efficiency)


#trovare il migliore lambda usando gurobi, prende la response, la truth e la reco, e printa l'andamento
def gurobi(response, reco, truth, efficiency): #sono array numpy 2d e 1d (la reco sarebbe la reco per la purity)
    chi2_array= np.zeros(20)
    lam_array = np.array([0, 0.00001, 0.00005, 0.0001, 0.0002, 0.0004, 0.0006, 0.0008, 0.001, 0.003, 0.005, 0.007, 0.009, 0.01, 0.03, 0.05, 0.1, 0.5, 1, 1.5])
    for i in range(20):  
        unfolder = QUnfoldQUBO(response, reco, lam_array[i])
        unfolder.initialize_qubo_model()
        unfolded, __, cov_matrix = unfolder.solve_gurobi_integer()
        unfolded=unfolded/efficiency
        chi2=compute_chi2(unfolded, truth, cov_matrix)
        chi2_array[i] = chi2

    plt.scatter(lam_array, chi2_array, alpha=0.5)
    plt.show()

    i = np.argmin(chi2_array) #restituisce l'indice dell'elemento più piccolo
    print(lam_array[i])
    return lam_array[i] #non restituisce però il valore del chi quadro minimo

    
#scelto numero di reads e numero di toys, fa N iterazioni e trova media e errore chi2
def get_chi2_sa(truth, reco, response, efficiency, n_reads, n_toys, lam, n=10):
    chi2_array = np.zeros(n)
    for i in range(n):
        unfolder = QUnfoldQUBO(response, reco, lam = lam)
        unfolder.initialize_qubo_model()
        unfolded, __, cov_matrix = unfolder.solve_simulated_annealing(num_reads=n_reads, num_toys=n_toys)
        unfolded = unfolded/efficiency
        chi2=compute_chi2(unfolded, truth, cov_matrix)
        chi2_array[i] = chi2
    mean_chi2 = np.mean(chi2_array)
    error_chi2 = np.std(chi2_array)/np.sqrt(n)
    return mean_chi2, error_chi2


#reads fisso, toys variabile, trova un nome
##########PER TESTARE USA MENO DATI#####################
def chi2_fixed_reads(truth, reco, response, efficiency, n_reads, lam):
    #toys_array=np.array([500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000])
    toys_array=np.array([10, 50, 80, 100, 130, 170, 200, 220, 250, 270, 300, 330, 370, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000])
    chi2_array=np.zeros(26)
    error_chi2_array=np.zeros(26)
    for i in range(26):
        mean_chi2, error_chi2 = get_chi2_sa(truth, reco, response, efficiency, n_reads, n_toys=toys_array[i], lam=lam, n=10)
        chi2_array[i] = mean_chi2
        error_chi2_array[i] = error_chi2
    #bisognerebbe mettere la progress bar
    #plottare
    fig = plt.figure()
    plt.errorbar(toys_array, chi2_array, yerr=error_chi2_array, label='Chi2 in function of n_toys, n_reads fixed=100')
    plt.legend()
    plt.show()


#toys fisso, reads variabile, trova un nome
##########PER TESTARE USA MENO DATI#####################
def chi2_fixed_toys(truth, reco, response, efficiency, n_toys, lam):
    reads_array=np.array([1, 10, 20, 30, 50, 60, 70, 80, 100, 120, 150, 180, 200, 230, 260, 300, 340, 370, 400, 440, 470, 500])
    chi2_array=np.zeros(22)
    error_chi2_array=np.zeros(22)
    for i in range(22):
        mean_chi2, error_chi2 = get_chi2_sa(truth, reco, response, efficiency, n_reads=reads_array[i], n_toys=n_toys, lam=lam, n=10)
        chi2_array[i] = mean_chi2
        error_chi2_array[i] = error_chi2
    #bisognerebbe mettere la progress bar
    #plottare
    fig = plt.figure()
    plt.errorbar(reads_array, chi2_array, yerr=error_chi2_array, label='Chi2 in function of n_reads, n_toys fixed=70')
    plt.legend()
    plt.show()
