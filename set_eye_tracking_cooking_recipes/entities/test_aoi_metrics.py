import os
import numpy as np
import pandas as pd
import csv
import re
from set_eye_tracking_cooking_recipes.entities.test_data import AoiIndexes


class TestAoiMetrics:
    testid: int
    aois_semmeln_total_duration_of_fixations: np.ndarray
    aois_schweinebraten_total_duration_of_fixations: np.ndarray
    aois_hawai_total_duration_of_fixations: np.ndarray
    aois_indi_curry_total_duration_of_fixations: np.ndarray
    aois_chilli_total_duration_of_fixations: np.ndarray
    aois_glasnudeln_total_duration_of_fixations: np.ndarray
    aois_gratine_total_duration_of_fixations: np.ndarray
    aois_spaghettis_total_duration_of_fixations: np.ndarray
    aois_arab_pizza_total_duration_of_fixations: np.ndarray
    aois_hack_total_duration_of_fixations: np.ndarray

    def __init__(self, file_path : str, id: int, Testperson: str):
        data = pd.read_csv(file_path, delimiter='\t')

        self.id = id
        self.aois_semmeln_total_duration_of_fixations = self.extract_data(data=data, keyword="semmeln", name_participant=Testperson)
        self.aois_schweinebraten_total_duration_of_fixations = self.extract_data(data=data, keyword="schweinebraten", name_participant=Testperson)
        self.aois_hawai_total_duration_of_fixations = self.extract_data(data=data, keyword="hawai", name_participant=Testperson)
        self.aois_indi_curry_total_duration_of_fixations = self.extract_data(data=data, keyword="indi_curry", name_participant=Testperson)
        self.aois_chilli_total_duration_of_fixations = self.extract_data(data=data, keyword="chilli", name_participant=Testperson)
        self.aois_glasnudeln_total_duration_of_fixations = self.extract_data(data=data, keyword="glasnudeln", name_participant=Testperson)
        self.aois_gratine_total_duration_of_fixations = self.extract_data(data=data, keyword="gratine", name_participant=Testperson)
        self.aois_spaghettis_total_duration_of_fixations = self.extract_data(data=data, keyword="spaghettis", name_participant=Testperson)
        self.aois_arab_pizza_total_duration_of_fixations = self.extract_data(data=data, keyword="arab_pizza", name_participant=Testperson)
        self.aois_hack_total_duration_of_fixations = self.extract_data(data=data, keyword="hack", name_participant=Testperson)

    
    def extract_data(self, data, keyword, name_participant) -> np.ndarray:
         # Filtere die Daten nach den angegebenen Bedingungen
        filtered_data = data.loc[(data['Participant'] == name_participant) & (data['TOI'] == keyword), ['Total_duration_of_fixations', 'AOI']]

        filtered_data_reordered = self.rearrange_data(data=filtered_data)

        # Konvertiere die extrahierten Werte in ein numpy.ndarray und gib sie zurück
        return np.array(filtered_data_reordered)
    
    def rearrange_data(self, data):
            reordered_data = [0] * len(AoiIndexes)  # Initialisiere eine Liste mit Nullen entsprechend der Anzahl der AOIs

            for index, row in data.iterrows():
                aoi = row['AOI']
                duration = row['Total_duration_of_fixations']
                
                if aoi in AoiIndexes.__members__:
                    idx = AoiIndexes[aoi].value
                    reordered_data[idx] = duration
        
            return reordered_data

    def calculate_mean(self):
        # Liste aller Arrays in der Klasse
        arrays = [getattr(self, attr) for attr in dir(self) if isinstance(getattr(self, attr), np.ndarray)]
        # Überprüfe, ob mindestens ein Array vorhanden ist
        if arrays:
            # Berechne den Mittelwert über alle Arrays
            mean_array = np.mean(arrays, axis=0)
            rounded_mean_array = np.round(mean_array).astype(int)  # Runde auf Integer
            return rounded_mean_array
        else:
            raise ValueError("Es sind keine Arrays vorhanden.")

