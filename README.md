# Milestone 2

## Installation
1. Make sure you have python 3.11 installed.
2. Have miniconda installed and configured https://docs.anaconda.com/free/miniconda/miniconda-install/
3. Run command below to install libraries required for this project
```
pip3 install -r requirements.txt
```

## Data
The data can be downloaded and setup in two ways:
1. If you are in a unix like operating system and want an automatic setup:
   navigate to the root of the repo and run `make`
2. If you are in windows or want to manually configure:
    - You can follow the below guide
    - you can download make and then run `make`

If you are following the manual config please follow the below steps:

We will store all project data and config files in a directory named Data, my directory is set up as follows:

       -- /path/to/milestone2
           -- Data
           -- Scripts

navigate to `/path/to/milestone2` then run `mkdir Data` then you can download the data to that directory
using the following commands:

```
curl https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/437802/utd19_u.csv -o Data/traffic.csv
curl https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/437802/detectors_public.csv -o Data/detectors.csv
curl https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/437802/links.csv -o Data/links.csv
```
You should now see:
    
    --/path/to/milestone2
        -- Data
            -- detectors.csv
            -- links.csv
            -- traffic.csv
        -- Scripts

You might notice that the traffic.csv is very large.
Because we will be mostly dealing with data from one city at a time, we can pre-filter the dataset and store one dataset per 
city. This will add a small bit of overhead to our system, but in return we can access and process data from a given city
much much faster. Asssuming you followed the above instructions and named your csvs: "traffic.csv", "detectors.csv", 
and "links.csv" and put all your datasets in one directory, you can follow the next instructions to digest the data into
city chunks:

```
python -m Scripts.Data.split_df_by_city Data
```

If you didn't name your csv's according to the above or want to see more documentation about the function you can run
```
python -m Scripts.Data.split_df_by_city -h
```

After running the above, you should see a new directory called UTD:

    -- path/to/milestone2
        -- Data
            --UTD
            -- ...
        -- Scripts

in UTD you will see a directory for each city in the dataset, if you expand a city you will see 
detector.csv, link.csv, and traffic.csv for each city.

Last we need to setup the config file and add some metadata to each city in the UTD
```
python -m Scripts.setup_config Data/UTD
python -m Scripts.Data.get_city_geo_coords Data/UTD
```
