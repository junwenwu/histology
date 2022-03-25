
# Store input arguments: <model_type> <input_directory> <output_directory> <device>
MODEL_TYPE=$1
INPUT_DIRECTORY=$2
OUTPUT_DIRECTORY=$3
DEVICE=$4

# Make sure that the output directory exists.
mkdir -p "$OUTPUT_DIRECTORY"
mkdir -p "$OUTPUT_DIRECTORY$MODEL_TYPE"

# Run the inference code
python python/run_inference.py -d $DEVICE \
                                -i $INPUT_DIRECTORY \
                                -o $OUTPUT_DIRECTORY$MODEL_TYPE \
                                -m $MODEL_TYPE
