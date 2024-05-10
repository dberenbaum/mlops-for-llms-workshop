set -eux

conda create -y -n mlops-for-llms-workshop python==3.11
conda init
eval "$('/opt/conda/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
conda activate mlops-for-llms-workshop
pip install --no-input -r requirements.txt
