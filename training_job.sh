
# Store input arguments: <output_directory> <input_directory>
INPUT_DIRECTORY=$1
OUTPUT_DIRECTORY=$2
EPOCHS=$3
DEVICE=$4
CORES=$5

# The default path for the job is the user's home directory,
#  change directory to where the files are.
if [ ! -d "./results/" ];then
   mkdir -p "./results"
fi
# Make sure that the output directory exists.
mkdir -p "$OUTPUT_DIRECTORY"

# Install Tensorflow 
python python/run_training.py -i $INPUT_DIRECTORY \
                               -o $OUTPUT_DIRECTORY \
                               -e $EPOCHS \
                               -d $DEVICE \
                               -c $CORES
