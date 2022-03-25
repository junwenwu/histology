
# Run the quantization script
python ./python/quantize.py

# Run the benchmark for Tensorflow model with Stock Tensorflow, oneDNN and OpenVINO integration with Tensorflow
./run_tensorflow_inference_all.sh CPU 2>/dev/null | grep "Throughput" 

# Run the benchmark_app for FP32 model
benchmark_app \
        -m ./models/ov/FP32/saved_model.xml 2>/dev/null | grep Throughput | xargs echo FP32

# Run the benchmark_app for INT8 model
benchmark_app \
        -m ./models/ov/INT8/saved_model.xml 2>/dev/null | grep Throughput | xargs echo INT8

