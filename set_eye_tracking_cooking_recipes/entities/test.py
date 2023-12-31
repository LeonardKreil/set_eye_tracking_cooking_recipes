import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import ConnectionPatch

from set_eye_tracking_cooking_recipes.entities.test_person import TestPerson
from set_eye_tracking_cooking_recipes.entities.test_data import TestData
from set_eye_tracking_cooking_recipes.entities.test_aoi_metrics import TestAoiMetrics
from set_eye_tracking_cooking_recipes.entities.test_data import AoiIndexes


stimuli = ['semmeln', 'schweinebraten', 'hawai', 'indi_curry', 'chilli', 'glasnudeln', 'gratine', 'spaghettis', 'arab_pizza', 'hack']

class Test:
    test_person : TestPerson
    test_data : TestData
    test_aoimetrics: TestAoiMetrics

    def __init__(self, test_person: TestPerson, test_data : TestData, test_aoimetrics : TestAoiMetrics) -> None:
        self.test_person = test_person
        self.test_data = test_data
        self.test_aoimetrics = test_aoimetrics

    @staticmethod
    def plot_mean_of_nutrition(list_of_tests:list):
        aoi_labels = [aoi.name for aoi in AoiIndexes]

        mean_arrays_pays_attention = []
        mean_arrays_pays_no_attention = []
        for test_class in list_of_tests:
                if test_class.test_person.attention_to_nutrition == True:  # Ändere das Attribut entsprechend deiner Klassenstruktur
                    mean_arrays_pays_attention.append(test_class.test_aoimetrics.calculate_mean())
                else:
                    mean_arrays_pays_no_attention.append(test_class.test_aoimetrics.calculate_mean())
        
        if mean_arrays_pays_attention:
            mean_attention = np.mean(mean_arrays_pays_attention, axis=0)
        if mean_arrays_pays_no_attention:
            mean_no_attention = np.mean(mean_arrays_pays_no_attention, axis=0)

        df_pays_attention = pd.DataFrame({'Total_duration_of_fixations': mean_attention}, index=aoi_labels)
        df_pays_no_attention = pd.DataFrame({'Total_duration_of_fixations': mean_no_attention}, index=aoi_labels)

        # Erstellen des Balkendiagramms
        plt.figure(figsize=(12, 6))

        # Subplot für df_meat_food
        plt.subplot(1, 2, 1)
        plt.ylim(0, 16)
        plt.bar(df_pays_attention.index, df_pays_attention['Total_duration_of_fixations']/1000, color='skyblue')
        plt.xlabel('Area of Interest (AOI)')
        plt.ylabel('Total duration of fixations (in seconds)')
        plt.title('Total fixation duration for nutrition-focused participants')

        plt.xticks(rotation=45)  # Rotieren der Beschriftungen für bessere Lesbarkeit
        plt.tight_layout()

        # Subplot für df_vegetarian_food
        plt.subplot(1, 2, 2)
        plt.ylim(0, 16)
        plt.bar(df_pays_no_attention.index, df_pays_no_attention['Total_duration_of_fixations']/1000, color='green')
        plt.xlabel('Area of Interest (AOI)')
        plt.ylabel('Total duration of fixations (in seconds)')
        plt.title('Total fixation duration for non nutrition-focused participants')

        plt.xticks(rotation=45)  # Rotieren der Beschriftungen für bessere Lesbarkeit
        plt.tight_layout()

        # Anzeigen der Diagramme
        plt.show()

    def plot_first_second_of_stimuli_visit(self):
        micros = 1_000_000
        img = mpimg.imread("../data/stimuli/schweinebraten.png")

        _, ax = plt.subplots(figsize=(img.shape[1]/100, img.shape[0]/100))
        ax.imshow(img, extent=[0, img.shape[1], 0, img.shape[0]])

        for i, stimulus in enumerate(stimuli):
            df_stimulus = self.test_data.get_stimuli_df(stimuli=stimulus)
            filtered_stimulus = df_stimulus[df_stimulus['Timestamps'] <= micros]
            inverted_y = img.shape[0] - filtered_stimulus["gaze_point_y"]
            
            color = plt.cm.viridis(i / len(stimuli))
            
            x = filtered_stimulus["gaze_point_x"]
            y = inverted_y
            ax.scatter(x, y, color=color, label=stimulus, s=20)

        ax.set_title(f'First Second of Test Person with ID {self.test_person.test_id}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.legend()

        plt.show()

    @staticmethod
    def plot_first_second_of_stimuli_visit_list(list_of_tests: list):
        micros = 400_000
        img = mpimg.imread("../data/stimuli/schweinebraten.png")
        
        num_tests = len(list_of_tests)
        rows = num_tests // 2 if num_tests % 2 == 0 else (num_tests // 2) + 1
        
        fig, axes = plt.subplots(rows, 2, figsize=(12, 6 * rows))
        plt.subplots_adjust(wspace=0.0, hspace=0.0)  # Einstellen der horizontalen und vertikalen Abstände zwischen den Subplots

        for test in list_of_tests:
            row = test.test_person.test_id // 2
            col = test.test_person.test_id % 2
            
            ax = axes[row, col]
            
            ax.imshow(img, extent=[0, img.shape[1], 0, img.shape[0]])

            for i, stimulus in enumerate(stimuli):
                df_stimulus = test.test_data.get_stimuli_df(stimuli=stimulus)
                filtered_stimulus = df_stimulus[df_stimulus['Timestamps'] <= micros]
                inverted_y = img.shape[0] - filtered_stimulus["gaze_point_y"]
                
                color = plt.cm.viridis(i / len(stimuli))
                
                x = filtered_stimulus["gaze_point_x"]
                y = inverted_y
                ax.scatter(x, y, color=color, label=stimulus, s=20)

            ax.set_title(f'Gaze points in first second of test person with ID {test.test_person.test_id}')  # assuming test_person is accessible here
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.legend()

        plt.tight_layout()
        plt.show()


    @staticmethod
    def plot_aoi_sequence(aoi_sequence: list):
        coordinates_of_aoi = [(610,520), (417,783), (319,850), (434,848), (677,850), (1185, 234), (1215, 570), (1300, 793)]

        total_visit_time_per_aoi = TestData.calculate_aoi_total_fixation_times([aoi_sequence])
        # Sample data
        x = []
        y = []
        point_sizes = []

        for aoi in aoi_sequence:
            x.append(coordinates_of_aoi[aoi[0]][0])
            y.append(coordinates_of_aoi[aoi[0]][1])
            point_sizes.append(total_visit_time_per_aoi[aoi[0]])


        img = mpimg.imread("../data/stimuli/schweinebraten_with_aois.png")

        _, ax = plt.subplots(figsize=(img.shape[1]/100, img.shape[0]/100))
        ax.imshow(img, extent=[0, img.shape[1], 0, img.shape[0]])
        inverted_y = img.shape[0] - np.array(y)
        # Create a scatter plot with point sizes based on the values in point_sizes
        ax.scatter(x, inverted_y, s=point_sizes, color='blue', alpha=1.0)

        def find_indices(lst, target_id):
            return [index for index, (id, _) in enumerate(lst) if id == target_id]

        def get_first_id(target_id, data_list):
            for index, (first_id, _) in enumerate(data_list):
                if first_id == target_id:
                    return index

        ids_in_aoi_times = {item[0] for item in aoi_sequence}
        # Add text labels above each point
        for aoi_id in ids_in_aoi_times:
            text_offset = point_sizes[get_first_id(aoi_id, aoi_sequence)]/50.0
            ax.text(x[get_first_id(aoi_id, aoi_sequence)], inverted_y[get_first_id(aoi_id, aoi_sequence)] + text_offset, f'{find_indices(aoi_sequence, aoi_id)}', color='green', ha='center', va='center', fontsize=25)


        for i in range(len(x) - 1):
            xyA = (x[i], inverted_y[i])
            xyB = (x[i + 1], inverted_y[i + 1])
            coordsA = "data"
            coordsB = "data"
            arrow = ConnectionPatch(xyA, xyB, coordsA, coordsB, arrowstyle='->', color='red', connectionstyle="arc3,rad=.5", mutation_scale=50, linewidth=3.0)
            ax.add_patch(arrow)

        # Add labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('AOI hit sequence')

        # Show the plot
        plt.show()

    @staticmethod
    def print_bar_plot_vegeatrian(list_of_tests:list, isvegetarian: bool):
        aoi_labels = [aoi.name for aoi in AoiIndexes]

        veggi_attributes = [
            "aois_semmeln_total_duration_of_fixations",
            "aois_chilli_total_duration_of_fixations",
            "aois_gratine_total_duration_of_fixations",
            "aois_spaghettis_total_duration_of_fixations",
            "aois_arab_pizza_total_duration_of_fixations"
        ]

        array_veggi = []
        array_meat = []
        for test_class in list_of_tests:
            if test_class.test_person.vegetarian == isvegetarian:
                for attr_name, attr_value in vars(test_class.test_aoimetrics).items():
                    if attr_name in veggi_attributes and isinstance(attr_value, np.ndarray): 
                        array_veggi.append(attr_value)
                    else:
                        if isinstance(attr_value, np.ndarray): 
                            array_meat.append(attr_value)
        
        if array_meat:
            overall_veggi = np.mean(array_veggi, axis=0)
        if array_veggi:
            overall_meat = np.mean(array_meat, axis=0)

        df_vegetraian_food = pd.DataFrame({'Total_duration_of_fixations': overall_veggi}, index=aoi_labels)
        df_meat_food = pd.DataFrame({'Total_duration_of_fixations': overall_meat}, index=aoi_labels)
        # Erstellen des Balkendiagramms
        plt.figure(figsize=(12, 6))

        # Subplot für df_meat_food
        plt.subplot(1, 2, 1)
        plt.ylim(0, 18)
        plt.bar(df_meat_food.index, df_meat_food['Total_duration_of_fixations']/1000, color='skyblue')
        plt.xlabel('Area of Interest (AOI)')
        plt.ylabel('Total duration of fixations (in seconds)')
        plt.title('Total duration of fixations for Meat-based food')

        plt.xticks(rotation=45)  # Rotieren der Beschriftungen für bessere Lesbarkeit
        plt.tight_layout()

        # Subplot für df_vegetarian_food
        plt.subplot(1, 2, 2)
        plt.ylim(0, 18)
        plt.bar(df_vegetraian_food.index, df_vegetraian_food['Total_duration_of_fixations']/1000, color='green')
        plt.xlabel('Area of Interest (AOI)')
        plt.ylabel('Total duration of fixations (in seconds)')
        plt.title('Total duration of fixations for vegetarian food')

        plt.xticks(rotation=45)  # Rotieren der Beschriftungen für bessere Lesbarkeit
        plt.tight_layout()

        # Anzeigen der Diagramme
        plt.show()

    @staticmethod
    def plot_bar_plot_goodvsbad_cooks(list_of_tests: [list, list]):

        aoi_labels = [aoi.name for aoi in AoiIndexes]

        mean_array_good_cooks = []
        mean_array_bad_cooks = []
        for test_class in list_of_tests:
                if test_class.test_person.cooking_experience == True:  # Ändere das Attribut entsprechend deiner Klassenstruktur
                    mean_array_good_cooks.append(test_class.test_aoimetrics.calculate_mean())
                else:
                    mean_array_bad_cooks.append(test_class.test_aoimetrics.calculate_mean())

        if mean_array_good_cooks:
            overall_good_cooks = np.mean(mean_array_good_cooks, axis=0)
        if mean_array_bad_cooks:
            overall_bad_cooks = np.mean(mean_array_bad_cooks, axis=0)

        df_good_cooks = pd.DataFrame({'Total_duration_of_fixations': overall_good_cooks}, index=aoi_labels)
        df_bad_cooks = pd.DataFrame({'Total_duration_of_fixations': overall_bad_cooks}, index=aoi_labels)
        # Erstellen des Balkendiagramms
        plt.figure(figsize=(12, 6))

        # Subplot für df_meat_food
        plt.subplot(1, 2, 1)
        plt.ylim(0, 16)
        plt.bar(df_good_cooks.index, (df_good_cooks['Total_duration_of_fixations'])/1000, color='skyblue')
        plt.xlabel('Area of Interest (AOI)')
        plt.ylabel('Total duration of fixations (in seconds)')
        plt.title('Total duration of fixations for skilled Cooks')

        plt.xticks(rotation=45)  # Rotieren der Beschriftungen für bessere Lesbarkeit
        plt.tight_layout()

        # Subplot für df_vegetarian_food
        plt.subplot(1, 2, 2)
        plt.ylim(0, 16)
        plt.bar(df_bad_cooks.index, df_bad_cooks['Total_duration_of_fixations']/1000, color='green')
        plt.xlabel('Area of Interest (AOI)')
        plt.ylabel('Total duration of fixations (in seconds)')
        plt.title('Total duration of fixations for less skilled Cooks')

        plt.xticks(rotation=45)  # Rotieren der Beschriftungen für bessere Lesbarkeit
        plt.tight_layout()

        # Anzeigen der Diagramme
        plt.show()

    @staticmethod
    def scarf_plot_out_of_aoi_sequence(aoi_sequences: [list, list]):

        colors = {
            AoiIndexes.picture.value: 'red',
            AoiIndexes.rating.value: 'brown',
            AoiIndexes.duration.value: 'grey',
            AoiIndexes.easiness.value: 'green',
            AoiIndexes.calories.value: 'orange',
            AoiIndexes.ingredients.value: 'purple',
            AoiIndexes.preparation_time.value: 'yellow',
            AoiIndexes.preparation_text.value: 'skyblue'
        }

        colors_for_legend = {
            AoiIndexes.picture.name: 'red',
            AoiIndexes.rating.name: 'blue',
            AoiIndexes.duration.name: 'grey',
            AoiIndexes.easiness.name: 'green',
            AoiIndexes.calories.name: 'orange',
            AoiIndexes.ingredients.name: 'purple',
            AoiIndexes.preparation_time.name: 'yellow',
            AoiIndexes.preparation_text.name: 'skyblue'
        }

        # Liste mit Lebensmittelnamen
        liste_slides = [
            "Smmelknödel",
            "Schweinebraten",
            "Hawaii-Auflauf",
            "Indisches-Curry",
            "Chilli sin carne",
            "Glasnudelsalat",
            "Kartoffelgratine",
            "Spaghetti aglio olio",
            "arabische Pizza",
            "Nudel-Hack Auflauf"
        ]

        num_plots = len(aoi_sequences)

        fig, axes = plt.subplots(num_plots, 1, figsize=(8, 3*num_plots))

        for i, (data, ax) in enumerate(zip(aoi_sequences, axes)):
            base_position = 0
            for color, length in data:
                ax.barh(0, length, left=base_position, color=colors[color], height=1)
                base_position += length
            max_length = max(sum(item[1] for item in sublist) for sublist in aoi_sequences)
            ax.set_xlim(0, max_length)  # Setze X-Achsen-Limit auf die maximale Länge in allen Daten
            ax.text(-0.1, 0.5, f"{liste_slides[i]}", va='center', ha='center', rotation=90, transform=ax.transAxes)
            ax.set_yticks([])

        legend_labels = [f"{key}" for key, value in colors_for_legend.items()]
        plt.legend(handles=[plt.Rectangle((0,0),1,1, color=colors[key]) for key in colors],
                labels=legend_labels,
                loc='upper right', 
                fontsize='small', 
                title='Color Legend')

        plt.tight_layout()
        plt.show()

    
    @staticmethod
    def scarf_plot_out_of_one_sequence(aoi_sequence: list):

        colors = {
            AoiIndexes.picture.value: 'red',
            AoiIndexes.rating.value: 'brown',
            AoiIndexes.duration.value: 'grey',
            AoiIndexes.easiness.value: 'green',
            AoiIndexes.calories.value: 'orange',
            AoiIndexes.ingredients.value: 'purple',
            AoiIndexes.preparation_time.value: 'yellow',
            AoiIndexes.preparation_text.value: 'skyblue'
        }

        colors_for_legend = {
            AoiIndexes.picture.name: 'red',
            AoiIndexes.rating.name: 'blue',
            AoiIndexes.duration.name: 'grey',
            AoiIndexes.easiness.name: 'green',
            AoiIndexes.calories.name: 'orange',
            AoiIndexes.ingredients.name: 'purple',
            AoiIndexes.preparation_time.name: 'yellow',
            AoiIndexes.preparation_text.name: 'skyblue'
        }

        num_plots = 1  # Es wird nur eine Sequenz dargestellt

        fig, axes = plt.subplots(num_plots, 1, figsize=(8, 3 * num_plots))

        base_position = 0
        for color, length in aoi_sequence:
            axes.barh(0, length, left=base_position, color=colors[color], height=1)
            base_position += length
        
        max_length = sum(length for _, length in aoi_sequence)
        axes.set_xlim(0, max_length)  # Setze X-Achsen-Limit auf die Gesamtlänge der Sequenz
        axes.set_yticks([])

        legend_labels = [f"{key}" for key, value in colors_for_legend.items()]
        plt.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=colors[key]) for key in colors],
                    labels=legend_labels,
                    loc='upper right',
                    fontsize='small',
                    title='Color Legend')

        plt.tight_layout()
        plt.show()