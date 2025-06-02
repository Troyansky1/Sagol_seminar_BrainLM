
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/ai_center/ai_users/gonyrosenman/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/ai_center/ai_users/gonyrosenman/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/home/ai_center/ai_users/gonyrosenman/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/ai_center/ai_users/gonyrosenman/miniconda3/bin:$PATH"
    fi
fi 
unset __conda_setup
# <<< conda initialize <<<
export WANDB_API_KEY=f88cfc152415d62aa82abbb1c93524e8ddd6bbc6
echo "WANDB API key was updated successfully"
export PATH=$HOME/openmpi/bin:$PATH
export LD_LIBRARY_PATH=$HOME/openmpi/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=$HOME/openmpi/lib:$LIBRARY_PATH
echo "added openmpi to PATH, updated LD_LIBRARY_PATH with openmpi and LIBRARY_PATH"
export NLTK_DATA=/home/ai_center/ai_users/gonyrosenman/NLTK
echo "added NLTK download dir"

