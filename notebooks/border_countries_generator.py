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



country_extractor = extractor.tag_content_extractor("country")
bordering_countries_extractor = extractor.tag_content_extractor("bordering_countries")
non_bordering_countries_extractor = extractor.tag_content_extractor("non_bordering_countries")
    
class BorderCountriesGenerator(curator.LLM):
    PROMPT : str = """
Give me a list of countries that border with {country} and a list of countries that doesn't border with {country}.

Only return each country name wrapped in <country>..</country> tag. Wrap the list of bordering countries in <bordering_countries>..</bordering_countries> tag and the list of non-bordering countries in <non_bordering_countries>..</non_bordering_countries> tag.
    """.strip()
    def prompt(self, input: dict) -> str:
        """Generate a prompt for the subsubject generator."""
        return self.PROMPT.format(country=input["country"],)

    def parse(self, input: dict, response: str) -> dict:
        """Parse the model response along with the input to the model into the desired output format.."""
        bordering_countries_ = bordering_countries_extractor(response)
        assert len(bordering_countries_) > 0
        bordering_countries_response = bordering_countries_[0].strip()
        if len(bordering_countries_response) == 0:
            bordering_countries = []
        else:
            bordering_countries_ = country_extractor(bordering_countries_response)
            assert len(bordering_countries_) > 0
            
            bordering_countries = [x.strip() for x in bordering_countries_]
        
        input["bordering_countries"] = bordering_countries
        
        
        non_bordering_countries_ = non_bordering_countries_extractor(response)
        assert len(non_bordering_countries_) > 0
        non_bordering_countries_response = non_bordering_countries_[0].strip()
        if len(non_bordering_countries_response) == 0:
            non_bordering_countries = []
        else:
            non_bordering_countries_ = country_extractor(non_bordering_countries_response)
            assert len(non_bordering_countries_) > 0
            
            non_bordering_countries = [x.strip() for x in non_bordering_countries_]
        input["non_bordering_countries"] = non_bordering_countries
        
        return {**input}
    


# fact_generator = CommonCitiesGenerator(model_name="gpt-4o")
# df = pd.DataFrame(countries, columns=["country"])

# dataset = Dataset.from_pandas(df)
# dataset = fact_generator(dataset)

# dataset.save_to_disk("/u/zliu/datastor1/KE-by-CP/data/debug_meta_train/country_data/common_cities_generation.hf",)
countries = list(io.load_json(f"{vars.DATA_DIR}/debug_meta_train/country_syn_data/country2continent.json"))

continent_generator = BorderCountriesGenerator(model_name="gpt-4o", backend_params={
    "max_requests_per_minute": 30_000,     
    "max_tokens_per_minute": 150_000_000,
    "require_all_responses": False
},)

df = pd.DataFrame(countries, columns=["country"])
dataset = Dataset.from_pandas(df)
continent_generator(dataset,).save_to_disk("/u/zliu/datastor1/KE-by-CP/data/debug_meta_train/country_syn_data/border_countries.hf",)


print()
