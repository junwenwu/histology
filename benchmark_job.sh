
# Run the benchmark_app for FP32 model
benchmark_app \
        -m ./models/ov/FP32/saved_model.xml -t 10 2>/dev/null | grep Throughput | xargs echo FP32

# Run the benchmark_app for INT8 model
benchmark_app \
        -m ./models/ov/INT8/saved_model.xml -t 10 2>/dev/null | grep Throughput | xargs echo INT8

