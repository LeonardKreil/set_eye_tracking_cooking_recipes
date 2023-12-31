from enum import Enum

import pandas as pd

class Gender(Enum):
    male = "male"
    female = "female"
    divers = "divers"


class CookingNumberWeek(Enum):
    rare = 1
    medium = 2
    often = 3


class CookingTimeMeal(Enum):
    little = 1
    medium = 2
    many = 3


class MealComplexity(Enum):
    easy = 1
    medium = 2
    complex = 3


class TestPerson:
    test_id: int
    name: str
    vegetarian: bool
    gender: Gender
    favorites: list
    cooking_number_week: CookingNumberWeek
    cooking_time_meal: CookingTimeMeal
    meal_complexity: MealComplexity
    attention_to_nutrition: bool

    def __init__(self, surveys_file_path: str, id: int) -> None:
        df = pd.read_csv(surveys_file_path, delimiter=';')
        
        # Extract the row with the corresponding ID
        person_data = df[df['test_id'] == id].iloc[0]

        # Assign values to class attributes
        self.test_id = person_data['test_id']
        self.name = person_data['name']
        self.vegetarian = person_data['vegetarian']
        self.gender = Gender(person_data['gender'])
        self.favorites = person_data['favorites'].split(',')
        self.cooking_number_week = CookingNumberWeek(person_data['cooking_number_week'])
        self.cooking_time_meal = CookingTimeMeal(person_data['cooking_time_meal'])
        self.meal_complexity = MealComplexity(person_data['meal_complexity'])
        self.attention_to_nutrition = person_data['attention_to_nutrition']

    def to_dict(self):
        return {
            'test_id': self.test_id,
            'name': self.name,
            'vegetarian': self.vegetarian,
            'gender': self.gender.value,
            'favorites': self.favorites,
            'cooking_number_week': self.cooking_number_week.name,
            'cooking_time_meal': self.cooking_time_meal.name,
            'meal_complexity': self.meal_complexity.name,
            'attention_to_nutrition': self.attention_to_nutrition
        }
      

    @property
    def cooking_experience(self) -> bool:
        cooking_experience_int = self.cooking_number_week.value + self.cooking_time_meal.value + self.meal_complexity.value
        if cooking_experience_int > 5:
            return True
        else:
            return False
