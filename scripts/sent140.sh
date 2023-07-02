# Command for T0 (t03b, t0pp -- we ran the latter on CPUs, another option you can use is the HuggingfaceAPI https://huggingface.co/inference-api)
python -m privacy.main \
    --dataset sent140 \
    --model t03b \
    --paradigm prompt \
    --split test \
    --batch_size 4 \
    --use_gpu 1 \
    --seed 0 \
    --client_subsample 0.025


# Command for Bi-Encoder
python -m privacy.main \
    --dataset sent140 \
    --model dpr \
    --paradigm similarity \
    --use_gpu 1 \
    --seed 0 \
    --client_subsample 0.025
    --split test


# Command for GPT Variations with "User Privacy" 