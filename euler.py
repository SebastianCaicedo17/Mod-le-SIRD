import matplotlib.pyplot as plt 
import numpy
import pandas as pd
import math
from scipy.optimize import approx_fprime


step = 0.01
nb_jours = 90
nb_points = int(nb_jours / step)

transmission = 0.375
guerison = 0.1325
mortalite = 0.012499999999999999 

def next_values (transmission, guerison, mortalite, step, nb_jours):
    time = [0]
    suceptible_infect = [0.9988935424701451]
    infect = [0.00319670189361468]
    retablis = [-0.002454650379593453]
    decedes = [-0.004621651074395652]

    for i in range (1, nb_points):
        new_time = time[-1] + step
        new_suceptible_infect = (-transmission * suceptible_infect[-1] * infect[-1])* step + suceptible_infect[-1]
        new_infect = (transmission * suceptible_infect[-1] * infect[-1] - guerison * infect[-1] - mortalite * infect[-1])* step + infect[-1]
        new_retablis = (guerison * infect[-1])* step + retablis[-1]
        new_decedes = (mortalite * infect[-1])* step + decedes[-1]

        time.append(new_time)
        suceptible_infect.append(new_suceptible_infect)
        infect.append(new_infect)
        retablis.append(new_retablis)
        decedes.append(new_decedes)

    time = numpy.array(time[::100])
    suceptible_infect = numpy.array(suceptible_infect[::100])
    infect = numpy.array(infect[::100])
    retablis = numpy.array(retablis[::100])
    decedes = numpy.array(decedes[::100])

    return time, suceptible_infect, infect, retablis, decedes


def plot_data(time, suceptible_infect, infect, retablis, decedes, real_suceptible, real_infect, real_retablis, real_decedes):
    plt.figure(figsize=(15, 6))
    plt.plot(time, suceptible_infect, "b:", label='suceptible_infects prédiction')
    plt.plot(time, infect, "r:", label='infects prédiction')
    plt.plot(time, retablis,"g:", label ='retablis prédictions')
    plt.plot(time, decedes,color = "orange",linestyle=":", label= 'décédés prédiction')
    plt.plot(time, real_suceptible, "b-", label ='real_suceptible_infects')
    plt.plot(time, real_infect, "r-", label= "real_inféctés")
    plt.plot(time, real_retablis, "g-", label ="real_rétablis")
    plt.plot(time, real_decedes, color ="orange", label= "real_décédes")

    plt.xlabel('Temps (Jours)')
    plt.ylabel('Predictions')
    plt.title('Modèle SIRD')
    plt.legend()
    plt.show()

def ground_truth(path):
    df = pd.read_csv(path)
    return {
        'jours': df['Jour'].tolist(),
        'susceptibles': df['Susceptibles'].tolist(),
        'infectes': df['Infectés'].tolist(),
        'retablis': df['Rétablis'].tolist(),
        'deces': df['Décès'].tolist()
    }

def rmse(model, reel):
    nb_echantillions = len(model)
    total_squared_errors = 0

    for index in range(nb_echantillions):
        total_squared_errors += (model[index] - reel[index]) ** 2
    
    return math.sqrt(total_squared_errors / nb_echantillions)


def grid_search(step, nb_jours, path):

    transmissions = numpy.linspace(0.25, 0.5, 5)
    guerisons = numpy.linspace(0.08, 0.15, 5)
    mortalites = numpy.linspace(0.005, 0.015, 5)

    donnes = ground_truth('sird_dataset.csv')
    real_susceptibles = donnnes['susceptibles']
    real_infectes = donnnes['infectes']
    real_retablis = donnnes['retablis']
    real_deces = donnnes['deces']

    best_transmissions, best_guerisons, best_mortalites = None, None, None
    best_rmse = float('inf')

    for transmission in (transmissions):
        for guerison in (guerisons):
            for mortalite in (mortalites):
                time, suceptible_infect, infect, retablis, decedes = next_values(transmission, guerison, mortalite, step, nb_jours)
                rmse_suceptible = rmse(suceptible_infect, real_susceptibles)
                rmse_infctes = rmse( infect, real_infectes)
                rmse_retablis = rmse(retablis, real_retablis)
                rmse_deces = rmse(decedes, real_deces)

                actual_rmse = rmse_suceptible + rmse_retablis + rmse_infctes + rmse_deces

                if actual_rmse < best_rmse:
                    best_rmse = actual_rmse
                    best_transmissions, best_guerisons, best_mortalites = transmission, guerison, mortalite

    return best_transmissions, best_guerisons, best_mortalites


if __name__ == "__main__":
    
    time, suceptible, infectes, retablis, deces = next_values (transmission, guerison, mortalite, step, nb_jours)
    donnnes= ground_truth('sird_dataset.csv')
    real_susceptibles = donnnes['susceptibles']
    real_infectes = donnnes['infectes']
    real_retablis = donnnes['retablis']
    real_deces = donnnes['deces']
    suceptible_rmse = rmse(suceptible, real_susceptibles)
    infectes_rmse = rmse(infectes, real_infectes)
    retablis_rmse = rmse ( retablis, real_retablis)
    decedes_rmse = rmse (deces, real_deces)
    print("Suceptible RMSE", suceptible_rmse)
    print("Infectes RMSE", infectes_rmse)
    print("Retablis RMSE", retablis_rmse)
    print("Decedes RMSE", decedes_rmse)
    plot_data(time, suceptible, infectes, retablis, deces ,real_susceptibles, real_infectes, real_retablis, real_deces)

    best_transmissions, best_guerisons, best_mortalites = grid_search(step, nb_jours, donnnes)
    print("Meilleur transmission", best_transmissions)
    print("Meilleur guerison",best_guerisons)
    print("Meilleur mortalite", best_mortalites)

    #plot_data(time, suceptible, infectes, retablis, deces ,real_susceptibles, real_infectes, real_retablis, real_deces)


