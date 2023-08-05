from datetime import datetime

from airflow import DAG
from airflow.operators.bash_operator import BashOperator

# Define the paths to the scripts and data files
diffusers_repo = 'https://github.com/huggingface/diffusers.git'
params_file = '/path/to/params.yaml'

# Define the default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 8, 3),
    'retries': 1,
}

# Create the DAG with the defined arguments
dag = DAG(
    'pokemon_data_pipeline',
    default_args=default_args,
    schedule_interval=None,
)

# Define the tasks in the DAG using BashOperator
set_up_diffusers_task = BashOperator(
    task_id='set_up_diffusers',
    bash_command=f'git clone --depth 1 --branch v0.14.0 {diffusers_repo} diffusers && '
                 f'pip3.10 install -r "diffusers/examples/dreambooth/requirements.txt" && '
                 f'accelerate config default',
    dag=dag,
)

scrape_pokemon_images_task = BashOperator(
    task_id='scrape_pokemon_images',
    bash_command=f'python3 src/scrape_pokemon_images.py --params {params_file}',
    dag=dag,
)

download_pokemon_stats_task = BashOperator(
    task_id='download_pokemon_stats',
    bash_command='kaggle datasets download -d brdata/complete-pokemon-dataset-gen-iiv -f Pokedex_Cleaned.csv -p data/external/',
    dag=dag,
)

resize_pokemon_images_task = BashOperator(
    task_id='resize_pokemon_images',
    bash_command=f'python3 src/resize_pokemon_images.py --params {params_file}',
    dag=dag,
)

train_lora_task = BashOperator(
    task_id='train_lora',
    bash_command=f'accelerate launch --mps "diffusers/examples/dreambooth/train_dreambooth_lora.py" '
                 f'--pretrained_model_name_or_path=${{params.train_lora.base_model}} '
                 f'--instance_data_dir=${{params.train_lora.data_etl.train_data_path}} '
                 f'--output_dir=${{params.train_lora.train_lora.lora_path}} '
                 f'--instance_prompt=\'a pkmnlora pokemon\' '
                 f'--resolution=512 '
                 f'--train_batch_size=1 '
                 f'--gradient_accumulation_steps=1 '
                 f'--checkpointing_steps=500 '
                 f'--learning_rate=${{params.train_lora.learning_rate}} '
                 f'--lr_scheduler=\'cosine\' '
                 f'--lr_warmup_steps=0 '
                 f'--max_train_steps=${{params.train_lora.max_train_steps}} '
                 f'--seed=${{params.train_lora.seed}}',
    dag=dag,
)

generate_text_to_image_task = BashOperator(
    task_id='generate_text_to_image',
    bash_command=f'python3 src/generate_text_to_image.py --params {params_file}',
    dag=dag,
)

# Set dependencies between tasks
set_up_diffusers_task >> scrape_pokemon_images_task
download_pokemon_stats_task >> resize_pokemon_images_task
scrape_pokemon_images_task >> resize_pokemon_images_task
resize_pokemon_images_task >> train_lora_task
train_lora_task >> generate_text_to_image_task
