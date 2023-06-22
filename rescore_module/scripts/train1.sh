PJ_DIR=/home/is/yuki-yama/work/d3/dep-forest-complex/rescore_module
DATA_DIR=${PJ_DIR}/data/wiki
OUT_DIR=${PJ_DIR}/models
SCRIPT_DIR=${PJ_DIR}/scripts

##data order (for naming the model)
data_type=1

##data path
train_data=${DATA_DIR}/train${data_type}.json
dev_data=${DATA_DIR}/dev${data_type}.json

##model
model_name=bert-base-uncased

##hyperpatameters
epochs=2
learning_rate=3e-5
batch_size=32

##save dir
train_model_name=${model_name}_${data_type}_${epochs}_${learning_rate}_${batch_size}
output=${OUT_DIR}/${train_model_name}
#output=${OUT_DIR}/test

mkdir -p ${output}

cd ${SCRIPT_DIR}

CUDA_VISIBLE_DEVICES=0 \
python ${SCRIPT_DIR}/main.py \
  --model_name_or_path ${model_name} \
  --train_file ${train_data} \
  --validation_file ${dev_data} \
  --per_device_train_batch_size ${batch_size} \
  --learning_rate ${learning_rate} \
  --num_train_epochs ${epochs} \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir ${output}