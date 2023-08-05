DIFFUSERS_REPO = https://github.com/huggingface/diffusers.git
PARAMS_FILE = /path/to/params.yaml

.PHONY: all set_up_diffusers scrape_pokemon_images download_pokemon_stats resize_pokemon_images train_lora generate_text_to_image

all: generate_text_to_image

set_up_diffusers:
	git clone --depth 1 --branch v0.14.0 $(DIFFUSERS_REPO) diffusers
	pip3.10 install -r "diffusers/examples/dreambooth/requirements.txt"
	accelerate config default

scrape_pokemon_images: set_up_diffusers
	python3 src/scrape_pokemon_images.py --params $(PARAMS_FILE)

download_pokemon_stats:
	kaggle datasets download -d brdata/complete-pokemon-dataset-gen-iiv -f Pokedex_Cleaned.csv -p data/external/

resize_pokemon_images: scrape_pokemon_images download_pokemon_stats
	python3 src/resize_pokemon_images.py --params $(PARAMS_FILE)

train_lora: set_up_diffusers resize_pokemon_images
	accelerate launch --mps "diffusers/examples/dreambooth/train_dreambooth_lora.py" \
	--pretrained_model_name_or_path=${train_lora.base_model} \
	--instance_data_dir=${data_etl.train_data_path} \
	--output_dir=${train_lora.lora_path} \
	--instance_prompt='a pkmnlora pokemon' \
	--resolution=512 \
	--train_batch_size=1 \
	--gradient_accumulation_steps=1 \
	--checkpointing_steps=500 \
	--learning_rate=${train_lora.learning_rate} \
	--lr_scheduler='cosine' \
	--lr_warmup_steps=0 \
	--max_train_steps=${train_lora.max_train_steps} \
	--seed=${train_lora.seed}

generate_text_to_image: train_lora
	python3 src/generate_text_to_image.py --params $(PARAMS_FILE)
