# # default example
# python ./src/data/prepare_data.py \
# --data-dir '/home/ning/data/NC2C_OxAAA_paired_iso/test/registered' \
# --label-dir '/home/ning/data/NC2C_OxAAA_paired_iso/test/labels' \
# --output-dir '/home/ning/data/NC2C_OxAAA_paired_160_160_128' \
# --subset 'test'

# For testing DDPM (not ldm) experimental
# python ./src/data/prepare_data_exp.py \
# --data-dir '/home/ning/hdd/DATASETS_DEV_NING/NC2C_OxAAA_paired_iso_flipped/test/noncontrast' \
# --output-dir '/home/ning/hdd/DATASETS_DEV_NING/NC2C_OxAAA_paired_iso_32t3_exp' \
# --subset 'test' \
# --image-size 32 32 32

python ./src/data/prepare_data_exp.py \
--data-dir '/home/ning/hdd/DATASETS_DEV_NING/NC2C_OxAAA_paired_iso_flipped/test/contrast' \
--output-dir '/home/ning/hdd/DATASETS_DEV_NING/NC2C_OxAAA_paired_iso_32t3_exp' \
--subset 'test' \
--image-size 32 32 32