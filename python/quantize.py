import os
import numpy as np

from addict import Dict
from compression.graph import load_model, save_model
from compression.api.data_loader import DataLoader
from compression.engines.ie_engine import IEEngine
from compression.pipeline.initializer import create_pipeline

def read_data(input_data):
    npzfiles = np.load(input_data, allow_pickle = True )
    array_01, array_02 = npzfiles.files
    img_batch = npzfiles[array_01]
    lbl_batch = npzfiles[array_02]
    images = np.vstack(img_batch)
    labels = np.hstack(lbl_batch)
    return(images, labels)

class DatasetsDataLoader(DataLoader):
 
    def __init__(self, config):
        super().__init__(config)
        self.images, self.labels = read_data(str(config['data_source']))

    @property
    def size(self):
        return len(self.images)

    def __len__(self):
        return self.size

    def __getitem__(self, item):
        image = self.images[item]
        label = self.labels[item]

        # Prepare input for model
        rgb_image = np.expand_dims(np.transpose(image, (2,0,1)), 0)

        return (item, None), rgb_image, label

work_directory=os.getcwd() #environ['WORK_DIR']

# Dictionary with the FP32 model info
model_config = Dict({
    'model_name': 'saved_model',
    'model': work_directory + '/models/ov/FP32/saved_model.xml',
    'weights': work_directory + '/models/ov/FP32/saved_model.bin',
})

# Dictionary with the engine parameters
engine_config = Dict({
    'device': 'CPU',
    'stat_requests_number': 4,
    'eval_requests_number': 4
})

dataset_config = Dict({
    'data_source': work_directory + '/test_data/testdata.npz', # Path to input data for quantization
})

# Quantization algorithm settings
algorithms = [
    {
        'name': 'DefaultQuantization', # Optimization algorithm name
        'params': {
            'target_device': 'CPU',
            'preset': 'performance', # Preset [performance (default), accuracy] which controls the quantization mode 
                                     # (symmetric and asymmetric respectively)
            'stat_subset_size': 300  # Size of subset to calculate activations statistics that can be used
                                     # for quantization parameters calculation.
        }
    }
]

# Load the model.
model = load_model(model_config)

# Initialize the data loader.
data_loader = DatasetsDataLoader(dataset_config)

# Initialize the engine for metric calculation and statistics collection.
engine = IEEngine(engine_config, data_loader, None)

# Create a pipeline of compression algorithms.
pipeline = create_pipeline(algorithms, engine)

# Execute the pipeline.
compressed_model = pipeline.run(model)

# Save the compressed model.
save_model(compressed_model, work_directory + '/models/ov/INT8')
