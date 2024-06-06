.PHONY: utd_config
.PHONY: save_city_coords
.PHONY: setup_utd

setup_utd: Data/UTD
	python -m Scripts.setup_config Data/UTD
	python -m Scripts.Data.get_city_geo_coords Data/UTD

Data/UTD: | Data/traffic.csv Data/links.csv Data/detectors.csv
	python -m Scripts.Data.split_df_by_city Data

Data/traffic.csv: | Data
	curl https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/437802/utd19_u.csv -o Data/traffic.csv

Data/links.csv: | Data
	curl https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/437802/links.csv -o Data/links.csv

Data/detectors.csv: | Data
	curl https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/437802/detectors_public.csv -o Data/detectors.csv

Data:
	mkdir Data