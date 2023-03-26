# Command for GPT Variations with "User Privacy" In context examples
for MODEL in gpt2.7 gpt1.3 gpt125m;  
do
    for NUM in 3 5; 
    do
        if [[ $MODEL == "gpt1.3" ]]; then
            BAT