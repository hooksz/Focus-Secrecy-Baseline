# Similarity search (zero shot) with bi-encoder model
python -m privacy.main \
    --dataset 20news  \
    --model dpr  \
    --paradigm similarity \
    --split test \
    --use_gpu 1 \
    --seed 0 


# Run zero-shot with GPT
for MODEL in g