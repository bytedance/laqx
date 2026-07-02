BASE_DIR="outputs/hofstadter/12_8_N8_pbc_V2/ace_small_N3e-1_fresh_kx0_ky4"

COMMON_ARGS=(
 --L1 12
 --L2 8
 --particles 8
 --particles_up 8
 --V 2
 --alpha 0.25
 --model hofstadter
 --dtype complex
 --network_name ace
 --boundary1 pbc
 --boundary2 pbc
 --use_x64
 --mcmc_step 60
 --ndet 1
 --hidden 128
 --layers 12
 --MLP_hidden 128
 --MLP_layers 1
 --seed 0
 --precision tf32
 --polarized
 --batchsize 4096
 --symmetry T
 --kx 0
 --ky 4
)

python main.py \
 --output ${BASE_DIR}\
 "${COMMON_ARGS[@]}" \
 --flux_theta 0\
 --steps 5000\
 --save_frequency 5000\
 --mode march\
 --norm 3e-1\
 --lr_start 1000\
 --lr0 4000\
 --reduce 100

echo ">>> Running initial inference for flux_theta = 0"
python main.py \
 --output ${BASE_DIR} \
 "${COMMON_ARGS[@]}" \
 --flux_theta 0 \
 --steps 2000 \
 --mode test \
 --obs polar \
 --reduce 0

PREV_DIR=${BASE_DIR}
PREV_STEPS=5000

for i in $(seq 1 10); do
    THETA=$(awk -v i=$i 'BEGIN {print i/10}')
    START_STEPS=${PREV_STEPS}
    END_STEPS=$((START_STEPS + 1000))
    
    CKPT_FILE=$(printf "ckpt_%06d.npz" ${START_STEPS})
    NEW_DIR="${BASE_DIR}_theta${THETA}"

    echo "================================================================"
    echo ">>> Starting Step: flux_theta = ${THETA}"
    echo ">>> Resuming from: ${CKPT_FILE}"
    echo ">>> Steps: ${START_STEPS} -> ${END_STEPS}"
    echo "================================================================"

    mkdir -p ${NEW_DIR}
    cp "${PREV_DIR}/${CKPT_FILE}" "${NEW_DIR}/${CKPT_FILE}"
    cp "${PREV_DIR}/log.csv" "${NEW_DIR}/log.csv"

    echo ">>> Running Fine-tuning for flux_theta = ${THETA}"
    python main.py \
     --output ${NEW_DIR} \
     "${COMMON_ARGS[@]}" \
     --flux_theta ${THETA} \
     --steps ${END_STEPS} \
     --save_frequency 1000 \
     --mode march \
     --norm 1e-2 \
     --lr_start ${START_STEPS} \
     --lr0 10000 \
     --reduce 100
     
    echo ">>> Running Polar Inference for flux_theta = ${THETA}"
    python main.py \
     --output ${NEW_DIR} \
     "${COMMON_ARGS[@]}" \
     --flux_theta ${THETA} \
     --steps 2000 \
     --mode test \
     --obs polar \
     --reduce 0

    PREV_DIR=${NEW_DIR}
    PREV_STEPS=${END_STEPS}
done