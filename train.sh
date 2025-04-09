set -e

for DATASET in GOODMotif2/basis GOODSST2/length LBAPcore/assay GOODCMNIST/color; do 
       for MODEL in GSAT LECI CIGA; do
              goodtg --config_path final_configs/${DATASET}/covariate/${MODEL}.yaml \
                     --seeds "1/2/3/4/5" \
                     --mitigation_sampling feat \
                     --task train \
                     --average_edge_attn mean \
                     --gpu_idx 1 \
                     # --mitigation_expl_scores hard \ # for HS
                     # --mitigation_readout weighted \ # for ER
                     # --model_name ${MODEL}GIN        # for LA
                     # --mitigation_sampling raw \    # for CF (CIGA only)
              echo "DONE TRAIN ${MODEL} ${DATASET}"
       done
done