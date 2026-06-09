# Transferring FWI Output to the Data Lake


This repository contains the scripts related to the Wildfire FWI use case of the Climate Adaptation Digital Twin (Climate DT). All the work is being developed in the frame of the [Destination Earth initiative](https://destination-earth.eu/) from the European Commission, where [ECMWF](https://destine.ecmwf.int/) is one of the Entrusted Entities.

LICENSE NOTE: the European Union, represented by the European Commission is the direct and sole owner of the intellectual property rights of these Results. 


## Step1:
python stac_hda_fwi.py



## Step2:

vi EO.FMI.DAT.DESTINE_CLIMATE_WILDFIRE_FWI/metadata/collection_config.json


Check File contin it has to be:  [-180.0, -89.9775, 180.0, 89.9775]

{
  "id": "EO.FMI.DAT.DESTINE_CLIMATE_WILDFIRE_FWI",
  "item_asset_ignore_list": [
    "item_config.json"
  ],
  "item_config_optional": true,
  "item_folder_level": "MM",
  "YYYY": "MM",
  "thumbnail_regex": "^thumbnail",
  "overview_regex": "^overview",
  "additional_property_keys": [
    "model",
    "experiment",
    "generation",
    "simulation_id"
  ],
  "bbox": [
    -180.0,
    -89.9775,
    180.0,
    89.9775
  ]
}


vi EO.FMI.DAT.DESTINE_CLIMATE_WILDFIRE_FWI/metadata/collection.json

  "stac_version": "1.1.0",
  "stac_extensions": [],
  "description":

## Step3:

python generate_item_metadata.py


## Step4: Cehck 
To check all is well with the tree

find EO.FMI.DAT.DESTINE_CLIMATE_WILDFIRE_FWI -maxdepth 4 -type d | sort

