from tensorflow import keras
from tensorflow.keras import layers

class ResidualBlock(layers.Layer):

    def __init__(self, filters, stride=1):
        super(ResidualBlock, self).__init__()
        
        # First convolution
        # Padding = same; implicitly handle zero padding
        self.conv1 = layers.Conv2D(filters, 3, strides=stride, padding='same', use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.relu1 = layers.ReLU()
        
        # Second convolution
        self.conv2 = layers.Conv2D(filters, 3, strides=1, padding='same', use_bias=False)
        self.bn2 = layers.BatchNormalization()
        
        # If spatial dimensions change and don't match
        if stride != 1:
            self.shortcut = keras.Sequential([
                layers.Conv2D(filters, 1, strides=stride, use_bias=False),
                layers.BatchNormalization()
            ])
        else:
            self.shortcut = lambda x: x
        
        # Activation function
        self.relu2 = layers.ReLU()
    
    def call(self, x, training=False):
        # Conv 1
        out = self.conv1(x)
        out = self.bn1(out, training=training)
        out = self.relu1(out)
        
        # Conv 2
        out = self.conv2(out)
        out = self.bn2(out, training=training)
        
        # Shortcut
        shortcut = self.shortcut(x)
        
        # Add and activiation function (relu)
        out = layers.Add()([out, shortcut])
        out = self.relu2(out)
        
        return out

def create_resnet18(num_classes=4, input_shape=(224, 224, 3)):
    """
    ResNet18 Model

    num_classes: Gleason Grading System splits into 4 categories
        1. Non-cancerous (NC)
        2. Grade 3 (G3)
        3. Grade 4 (G4)
        4. Grade 5 (G5)

    input_shape: Standard default 224x224 size with 3 RGB channels for colour images

    Input -> Stage 1 (conv, batch norm, relu, max pool) -> Stage 2 (2x convlution w/ 64 filters) -> Stage 3 ()
    """

    inputs = keras.Input(shape=input_shape)
    
    # Stage 1 - Initial Layer
    # - Extract low-level features such as edges, basic shapes
    # - Downsample and normalize to reduce computational overhead
    # - Retain only the most prominent features
    x = layers.Conv2D(64, 7, strides=2, padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(3, strides=2, padding='same')(x)
    
    # Stage 2 - 64 filters
    # - Learn abstract representation of cell features in WSI
    # - Capture patterns
    x = ResidualBlock(64, stride=1)(x)
    x = ResidualBlock(64, stride=1)(x)
    
    # Stage 3 - 128 filters
    # - Downsample
    # - Learn mid-level features from WSI
    x = ResidualBlock(128, stride=2)(x)
    x = ResidualBlock(128, stride=1)(x)
    
    # Stage 4 - 256 filters
    # - Downsample further
    # - Learn higher level feature representations and larger-scale patterns
    # - Critical part for distingushing gleason grades
    x = ResidualBlock(256, stride=2)(x)
    x = ResidualBlock(256, stride=1)(x)
    
    # Stage 5 - 512 filters
    # - Learn most abstract, high-level features
    # - Directly inform final classification
    x = ResidualBlock(512, stride=2)(x)
    x = ResidualBlock(512, stride=1)(x)
    
    # Stage 6 - Output Layer
    # - Reduce to 512-demensional vector
    # - Each 512 value represents average activation of that feature across the entire image
    # - Summary of all learned features
    # - Maps 512 features to 4 class probabilties with proper probability distribution (Softmax)
    # - Outputs to P(NC), P(G3), P(G4), P(G5) with highest probability informing gleason grade prediction
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name='resnet18')
    
    return model

def compile_model(model, learning_rate=1e-4):

    # Compile with optimizations, loss, and our required metrics
    # ADAM = Adaptive Moment Estimation (https://arxiv.org/abs/1412.6980)
    # Categorical crossentropy measures difference between predicted probability distribution and true label (one hot encoding)
    # Accuracy = percentage of predictions match true labels
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

if __name__ == "__main__":
    
    model = create_resnet18(num_classes=4)
    model = compile_model(model)
    
    model.summary()