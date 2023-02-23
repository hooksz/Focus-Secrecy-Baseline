# Similarity search (zero shot) with bi-encoder model
python -m privacy.main \
    --dataset 20news  \
    --model dpr  \
    --paradigm similarity \
    --split test \
    --use_gpu 1 \
    --seed 0 


# Run zero-shot with GPT
for MODEL in gpt125m gpt1.3 gpt2.7;  
do
    for NUM in 0; 
    do
        BATCH_SIZE=4
        if [[ $MODEL == "gpt1.3" ]]; then
            BATCH_SIZE=16
        elif [[ $MODEL == "gpt2.7" ]]; then 
            BATCH_SIZE=8
        e