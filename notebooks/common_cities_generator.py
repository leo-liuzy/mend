from datasets import load_dataset
from tqdm import tqdm
from knowledge_propagation.utils import vars, io, extractor
from scipy.stats import describe
from typing import List, Dict
import re
from copy import deepcopy
import pandas as pd

from bespokelabs import curator
from datasets import Dataset

countries = [
    "United States",
    "Canada",
    "United Kingdom",
    "Germany",
    "France",
    "Italy",
    "Spain",
    "Portugal",
    "Netherlands",
    "Belgium",
    "Switzerland",
    "Austria",
    "Sweden",
    "Norway",
    "Denmark",
    "Finland",
    "Russia",
    "China",
    "Japan",
    "South Korea",
    "India",
    "Brazil",
    "Mexico",
    "Argentina",
    "Chile",
    "Colombia",
    "South Africa",
    "Egypt",
    "Turkey",
    "Saudi Arabia",
    "United Arab Emirates",
    "Thailand",
    "Vietnam",
    "Indonesia",
    "Malaysia",
    "Singapore",
    "Australia",
    "New Zealand",
    "Greece",
    "Poland",
    "Ukraine",
    "Ireland",
    "Czech Republic",
    "Hungary",
    "Romania",
    "Israel",
    "Pakistan",
    "Philippines",
    "Bangladesh",
    "Nigeria",
    "Kenya"
]


city_extractor = extractor.tag_content_extractor("city")
continent_extractor = extractor.tag_content_extractor("continent")
    
class CommonCitiesGenerator(curator.LLM):
    PROMPT : str = """
Give me a long list of commonly known cities in {country}.

Only return each city name wrapped in <city>..</city> tag.
    """.strip()
    def prompt(self, input: dict) -> str:
        """Generate a prompt for the subsubject generator."""
        return self.PROMPT.format(country=input["country"],)

    def parse(self, input: dict, response: str) -> dict:
        """Parse the model response along with the input to the model into the desired output format.."""
        cities = city_extractor(response)
        assert len(cities) > 0
        cities = [x.strip() for x in cities]
            
        input["cities"] = cities
        return {**input}
    
    
class ContinentGenerator(curator.LLM):
    PROMPT : str = """
Which continent is {country} located in?

Only return continent name wrapped in <continent>..</continent> tag.
    """.strip()
    def prompt(self, input: dict) -> str:
        """Generate a prompt for the subsubject generator."""
        return self.PROMPT.format(country=input["country"],)

    def parse(self, input: dict, response: str) -> dict:
        """Parse the model response along with the input to the model into the desired output format.."""
        continent_ = continent_extractor(response)
        assert len(continent_) == 1
        continent = continent_[0].strip()
        input["continent"] = continent
        
        return {**input}
    


# fact_generator = CommonCitiesGenerator(model_name="gpt-4o")
# df = pd.DataFrame(countries, columns=["country"])

# dataset = Dataset.from_pandas(df)
# dataset = fact_generator(dataset)

# dataset.save_to_disk("/u/zliu/datastor1/KE-by-CP/data/debug_meta_train/country_data/common_cities_generation.hf",)

continent_generator = ContinentGenerator(model_name="gpt-4o")
country_train_data = io.load_jsonlines(f"{vars.DATA_DIR}/debug_meta_train/common_country_data/train.jsonl")
city_country_pairs = [d["(city, country)"] for d in country_train_data]
common_countries = list(set([cc[1] for cc in city_country_pairs]))
df = pd.DataFrame(common_countries, columns=["country"])
dataset = Dataset.from_pandas(df)
continent_generator(dataset).save_to_disk("/u/zliu/datastor1/KE-by-CP/data/debug_meta_train/common_country_data/continent_generation.hf",)


print()
