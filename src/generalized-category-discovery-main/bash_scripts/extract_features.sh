PYTHON='/opt/conda/bin/python'

hostname
nvidia-smi

export CUDA_VISIBLE_DEVICES=0

${PYTHON} -m methods.clustering.extract_features --dataset cifar100 --use_best_model 'True' \
 --warmup_model_dir '/home/padl22t4/padl/src/generalized-category-discovery-main/osr_novel_categories/metric_learn_gcd/log/(20.06.2022_|_22.105)/checkpoints/model.pt'


 