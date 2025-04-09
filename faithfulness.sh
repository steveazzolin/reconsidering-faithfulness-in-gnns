set -e

for DATASET in GOODMotif2/basis GOODSST2/length GOODCMNIST/color LBAPcore/assay; do
       for MODEL in LECI CIGA GSAT; do
              goodtg --config_path final_configs/${DATASET}/covariate/${MODEL}.yaml \
                     --seeds "1/2/3/4/5" \
                     --task eval_metric \
                     --metrics "suff++/nec" \
                     --average_edge_attn mean \
                     --mitigation_sampling feat \
                     --gpu_idx 0 \
                     --mask  \
                     --debias \
                     --samplingtype deconfounded \
                     --nec_number_samples prop_G_dataset \
                     --save_metrics \
                     --log_id faithfulness
              echo "DONE ${MODEL} ${DATASET}"
       done
done