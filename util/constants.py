from pathlib import Path

# Grand Challenge folders were input files can be found
ROOT = Path(rf'D:\jassorRepository\OCELOT_Dataset\jassor\test_docker')
GC_CELL_FPATH = ROOT / Path("/input/images/cell_patches/")
GC_TISSUE_FPATH = ROOT / Path("/input/images/tissue_patches/")

GC_METADATA_FPATH = ROOT / Path("/input/metadata.json")

# Grand Challenge output file
GC_DETECTION_OUTPUT_PATH = ROOT / Path("/output/cell_classification.json")

# Sample dimensions
SAMPLE_SHAPE = (1024, 1024, 3)

