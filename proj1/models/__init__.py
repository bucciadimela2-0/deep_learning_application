from .mlp import MLP
from .cnn import CNN
from typing import Tuple, List, Union, Optional, Any, Dict
import warnings


#Factory function to create neural network models.
def get_model(model_name: str, 
             #input_shape: Shape of input data (channels, height, width) or (features,)
              input_shape: Tuple[int, ...] = None,
              num_classes: int = None, 
            #**kwargs: Model-specific parameters
              **kwargs) -> Union[MLP, CNN]:
    
    # Validate common parameters
    if input_shape is None:
        raise ValueError("input_shape must be provided")
    if num_classes is None or num_classes <= 0:
        raise ValueError("num_classes must be a positive integer")
    
    model_name = model_name.lower().strip()
    supported_models = ["mlp", "cnn"]
    
    if model_name not in supported_models:
        raise ValueError(f"Model '{model_name}' not supported. "
                        f"Available models: {supported_models}")
    
    # Validate input_shape format
    if not isinstance(input_shape, (list, tuple)) or len(input_shape) == 0:
        raise ValueError("input_shape must be a non-empty tuple or list")
    
    if model_name == "mlp":
        return _create_mlp(input_shape, num_classes, **kwargs)
    elif model_name == "cnn":
        return _create_cnn(input_shape, num_classes, **kwargs)
    
#Create MLP model with automatic layer size configuration
def _create_mlp(input_shape: Tuple[int, ...], num_classes: int, **kwargs) -> MLP:
    
    # Calculate input dimension
    if len(input_shape) == 1:
        input_dim = input_shape[0]
    elif len(input_shape) == 3:  # (C, H, W)
        input_dim = input_shape[0] * input_shape[1] * input_shape[2]
    else:
        raise ValueError(f"MLP input_shape must be (features,) or (channels, height, width), "
                        f"got {input_shape}")
    
    # Get hidden layer configuration
    hidden_layers = kwargs.get("layer_sizes", kwargs.get("hidden_layers", [128]))
    
    if not isinstance(hidden_layers, (list, tuple)):
        raise ValueError("layer_sizes/hidden_layers must be a list of hidden layer dimensions, "
                        "e.g., [256, 128, 64]")
    
    if len(hidden_layers) == 0:
        warnings.warn("No hidden layers specified, creating direct input->output mapping")
    
    # Validate hidden layer dimensions
    for i, size in enumerate(hidden_layers):
        if not isinstance(size, int) or size <= 0:
            raise ValueError(f"Hidden layer {i} size must be a positive integer, got {size}")
    
    # Build complete layer configuration: input -> hidden layers -> output
    complete_layer_sizes = [input_dim] + list(hidden_layers) + [num_classes]
    
    # Extract MLP-specific parameters
    mlp_params = {
        'layer_sizes': complete_layer_sizes,
        'activation': kwargs.get('activation', 'relu'),
        'dropout_rate': kwargs.get('dropout_rate', 0.0),
        'batch_norm': kwargs.get('batch_norm', False),
        'name': kwargs.get('name', 'mlp')
    }
    
    return MLP(**mlp_params)

#Create CNN model
def _create_cnn(input_shape: Tuple[int, ...], num_classes: int, **kwargs) -> CNN:
        if len(input_shape) != 3:
            raise ValueError(f"CNN input_shape must be (channels, height, width), got {input_shape}")

        channels, height, width = input_shape
        # Extract CNN-specific parameters and instantiate it
        return CNN(
            input_channels=channels,
            input_size=(height, width),
            num_classes=num_classes,
            num_filters=kwargs.get("num_filters", 64),
            num_blocks=kwargs.get("num_blocks", 1),
            dropout_rate=kwargs.get("dropout_rate", 0.2),
            use_batch_norm=kwargs.get("use_batch_norm", True),
            skip=kwargs.get("skip", False)
            
        )




# Return list of supported model names.
def list_supported_models() -> List[str]:
   
    return ["mlp", "cnn", "resnet"]

#Get information about a specific model type.
def get_model_info(model_name: str) -> Dict[str, Any]:

    model_name = model_name.lower().strip()
    
    model_info = {
        "mlp": {
            "description": "Multi-Layer Perceptron - fully connected neural network",
            "input_format": "(features,) or (channels, height, width)",
            "parameters": {
                "layer_sizes/hidden_layers": "List of hidden layer dimensions [128, 64]",
                "activation": "Activation function ('relu', 'gelu', 'tanh', etc.)",
                "dropout_rate": "Dropout probability (0.0-1.0)",
                "batch_norm": "Whether to use batch normalization (bool)",
                "name": "Model name (str)"
            }
        },
        "cnn": {
            "description": "Convolutional Neural Network - for image processing",
            "input_format": "(channels, height, width)",
            "parameters": {
                "channels": "List of conv layer channels [32, 64, 128]",
                "kernel_size": "Convolution kernel size (int)",
                "pool_size": "Max pooling size (int)",
                "fc_size": "Fully connected layer size (int)",
                "dropout_rate": "Dropout probability (0.0-1.0)",
                "name": "Model name (str)"
            }
        }
    }
    
    if model_name not in model_info:
        raise ValueError(f"Model '{model_name}' not supported. "
                        f"Available: {list(model_info.keys())}")
    
    return model_info[model_name]

