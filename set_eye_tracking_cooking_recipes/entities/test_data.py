import os
import numpy as np
import pandas as pd
import csv
import re

from enum import Enum

class AoiIndexes(Enum):
    picture = 0
    rating = 1
    duration = 2
    easiness = 3
    calories = 4
    ingredients = 5
    preparation_time = 6
    preparation_text = 7

class TestData:
    test_id: int
    raw: pd.DataFrame
    recording_timestamp: np.ndarray
    recording_resolution_height: np.ndarray
    recording_resolution_width: np.ndarray
    gaze_point_x: np.ndarray
    gaze_point_y: np.ndarray
    presented_stimulus_name: np.ndarray
    presented_media_height: np.ndarray
    presented_media_width: np.ndarray
    eye_movement_type: np.ndarray
    fixation_point_x: np.ndarray
    fixation_point_y: np.ndarray
    aois_semmeln: np.ndarray
    aois_schweinebraten: np.ndarray
    aois_hawai: np.ndarray
    aois_indi_curry: np.ndarray
    aois_chilli: np.ndarray
    aois_glasnudeln: np.ndarray
    aois_gratine: np.ndarray
    aois_spaghettis: np.ndarray
    aois_arab_pizza: np.ndarray
    aois_hack: np.ndarray

    def __init__(self, file_path : str):
        df = pd.read_csv(file_path, delimiter='\t')
        self.raw = df
        df = self.remove_rows_not_in_dacs(df)

        self.id = int(os.path.basename(file_path).split('_')[0])
        self.recording_timestamp = df["Recording timestamp"].to_numpy()
        self.recording_resolution_height = df["Recording resolution height"].to_numpy()
        self.recording_resolution_width = df["Recording resolution width"].to_numpy()
        self.gaze_point_x = df["Gaze point X"].to_numpy()
        self.gaze_point_y = df["Gaze point Y"].to_numpy()
        self.presented_stimulus_name = df["Presented Stimulus name"].to_numpy()
        self.presented_media_height = df["Presented Media width"].to_numpy()
        self.presented_media_width = df["Presented Media height"].to_numpy()
        self.eye_movement_type = df["Eye movement type"].to_numpy()
        self.fixation_point_x = df["Fixation point X"].to_numpy()
        self.fixation_point_y = df["Fixation point Y"].to_numpy()
        self.aois_semmeln = self.extract_aoi_data(df, "semmel")
        self.aois_indi_curry = self.extract_aoi_data(df, "curry")
        self.aois_arab_pizza = self.extract_aoi_data(df, "pizza")
        self.aois_chilli = self.extract_aoi_data(df, "chili")
        self.aois_glasnudeln = self.extract_aoi_data(df, "glasnudel")
        self.aois_gratine = self.extract_aoi_data(df, "gratin")
        self.aois_hack = self.extract_aoi_data(df, "hack")
        self.aois_hawai = self.extract_aoi_data(df, "hawai")
        self.aois_schweinebraten = self.extract_aoi_data(df, "braten")
        self.aois_spaghettis = self.extract_aoi_data(df, "spaghetti")


    def extract_aoi_data(self, df, keyword) -> np.ndarray:
        order_dict = {item.name: item.value for item in AoiIndexes}

        def get_variable_name(element):
            return element.split(' - ')[1][:-1]

        pattern = re.compile(rf'.*{re.escape(keyword)}.*', re.IGNORECASE)  

        columns_matching_pattern = [col for col in df.columns if pattern.search(col)]
        sorted_columns_matching_pattern = sorted(columns_matching_pattern, key=lambda x: order_dict[get_variable_name(x)])
        selected_rows : pd.DataFrame = df[df.columns[df.columns.isin(columns_matching_pattern)]]

        df_reordered = selected_rows[sorted_columns_matching_pattern]
        return df_reordered.to_numpy()
        

    def remove_rows_not_in_dacs(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[(df["Gaze point X"].between(0, 1920)) & (df["Gaze point Y"].between(0, 1080))]
    
    def get_stimuli_df(self, stimuli: str) -> pd.DataFrame:
        stimuli_indices = np.where(self.presented_stimulus_name == stimuli)[0]

        timestamps = self.recording_timestamp[stimuli_indices]
        min_timestamp = np.min(timestamps)
        adjusted_timestamps = timestamps - min_timestamp

        gaze_point_x = self.gaze_point_x[stimuli_indices]
        gaze_point_y = self.gaze_point_y[stimuli_indices]

        return pd.DataFrame({'Timestamps': adjusted_timestamps, 'gaze_point_x': gaze_point_x, 'gaze_point_y': gaze_point_y})
    
    def get_aoi_times(self) -> [list, list]:
        aoi_sequences = []
        aoi_total_fixation_times = []

        aois = [self.aois_semmeln, self.aois_indi_curry, self.aois_arab_pizza, self.aois_chilli, 
                self.aois_glasnudeln, self.aois_gratine, self.aois_hack, self.aois_hawai, 
                self.aois_schweinebraten, self.aois_spaghettis]


        for aoi in aois:
            rows_with_one = np.any(aoi == 1, axis=1)

            filtered_array = aoi[rows_with_one]

            previous_aoi_id = np.argmax(filtered_array[0,:])
            counter = 0

            current_aoi_times = []

            for row in filtered_array:
                current_aoi_id = np.argmax(row)
                if current_aoi_id == previous_aoi_id:
                    counter = counter + 1
                else:
                    current_aoi_times.append((previous_aoi_id, counter))

                    previous_aoi_id = current_aoi_id
                    counter = 0
            current_aoi_times.append((current_aoi_id, counter))

            total_fixation_times = [0] * (max(idx for idx, _ in current_aoi_times) + 1)

            for index, value in current_aoi_times:
                total_fixation_times[index] += value

            aoi_sequences.append(current_aoi_times)
            aoi_total_fixation_times.append(total_fixation_times)

        return aoi_sequences, aoi_total_fixation_times
    
    @staticmethod
    def calculate_aoi_total_fixation_times(aoi_sequences: list) -> np.ndarray:
        number_of_sequences = len(aoi_sequences)

        total_fixation_times = np.zeros((8,))
        for aoi_sequence in aoi_sequences:
            for index, value in aoi_sequence:
                total_fixation_times[index] += value

        return total_fixation_times / number_of_sequences

    @staticmethod
    def create_follower_array(aoi_sequences : [list, list], id: int) -> np.ndarray:
        follower_array = np.zeros((8,))
        for aoi_sequence in aoi_sequences:
            for i,t in enumerate(aoi_sequence):
                if t[0] == id and i < len(aoi_sequence) - 1:
                    follower_array[aoi_sequence[i+1][0]] += 1

        return follower_array
    
    @staticmethod
    def make_average_aoi_sequence(aoi_sequences: list) -> list:
        average_aoi_sequence = []
        average_length = int((sum(len(sequence) for sequence in aoi_sequences) / len(aoi_sequences))*1.25)

        # Starting point is the point which was the starting point most of the times     
        starting_aoi_id_occurrence = np.zeros((8,))
        for sequence in aoi_sequences:
            starting_aoi_id_occurrence[sequence[0][0]] += 1

        starting_aoi_id = np.argmax(starting_aoi_id_occurrence)

        average_aoi_sequence.append((starting_aoi_id,))
        current_id = starting_aoi_id

        def is_id_already_in_sequence(sequence, id) -> bool:
            counter = 0
            for tup in sequence:
                if tup[0] == id:
                    counter += 1

            if counter < 2:
                return False
            return True


        for _ in range(1, average_length):
            follower_array = TestData.create_follower_array(aoi_sequences, current_id)
            sorted_indices = np.argsort(follower_array)[::-1]
            for i in range(len(sorted_indices)):
                follower_id = sorted_indices[i]
                if(is_id_already_in_sequence(average_aoi_sequence, follower_id)):
                    pass
                else:
                    average_aoi_sequence.append((follower_id,))
                    break
            current_id = follower_id

        aoi_total_fixation_times = TestData.calculate_aoi_total_fixation_times(aoi_sequences)

        for i, t in enumerate(average_aoi_sequence):
            average_aoi_sequence[i] = (t[0], aoi_total_fixation_times[t[0]])

        # delete nulls
        return [(x, y) for x, y in average_aoi_sequence if y != 0.0]



    
    








