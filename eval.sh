set -e

# see *train.sh* for enabling mitigation strategies

for DATASET in GOODMotif2/basis LBAPcore/assay GOODSST2/length GOODCMNIST/color; do 
       for MODEL in LECI GSAT CIGA; do
              goodtg --config_path final_configs/${DATASET}/covariate/${MODEL}.yaml \
                     --seeds "1/2/3/4/5" \
                     --mitigation_sampling feat \
                     --task test \
                     --average_edge_attn mean \
                     --gpu_idx 1
              echo "DONE EVAL ${MODEL} ${DATASET}"
       done
done