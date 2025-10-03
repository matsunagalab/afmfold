import torch

from e2cnn import gspaces
from e2cnn import nn

class CNSteerableCNN(torch.nn.Module):
    def __init__(self, N, image_shape, channels=[3,6,6,12,12,8], kernel_size=[7,5,5,5,5,5], num_hidden_layers=3, hidden_dim=64, output_dim=64):
        super().__init__()
        self.N = N
        self.image_shape = image_shape
        self.channels = channels
        self.kernel_size = kernel_size
        self.num_hidden_layers = num_hidden_layers
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Get image shape
        *_, image_size, _image_size = image_shape
        assert image_size == _image_size, image_shape
        assert image_size % 2 == 1, image_shape
        self.image_size = image_size
        self.image_radius = int(image_size / 2)
        
        # the model is equivariant under rotations by 360/N degrees, modelled by C_N
        self.r2_act = gspaces.Rot2dOnR2(N=N)
        
        # Define convolutional layers
        assert len(channels) > 1, channels
        assert len(kernel_size) == len(channels)
        self.num_layers = len(channels)
        e2cnn_layers = []
        mlp_layers = []
        hidden_image_size = []
        
        # Define the first convolutional layer
        # Specify input/output types
        in_type = nn.FieldType(self.r2_act, [self.r2_act.trivial_repr])
        out_type = nn.FieldType(self.r2_act, channels[0]*N*[self.r2_act.regular_repr])
        self.input_type = in_type
        
        block1 = nn.SequentialModule(
            nn.MaskModule(in_type, 2*self.image_radius+1, margin=1),
            nn.R2Conv(in_type, out_type, kernel_size=kernel_size[0], padding=int(kernel_size[0]/2), bias=False),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True)
        )
        e2cnn_layers.append(block1)
        hidden_image_size.append({"in": image_size, "out": image_size})
        
        # Define intermediate layers
        for i in range(self.num_layers-1):
            in_type = e2cnn_layers[-1].out_type
            out_type = nn.FieldType(self.r2_act, channels[1+i]*N*[self.r2_act.regular_repr])
            
            block_i = nn.SequentialModule(
                nn.R2Conv(in_type, out_type, kernel_size=kernel_size[1+i], padding=int(kernel_size[1+i]/2), bias=False),
                nn.InnerBatchNorm(out_type),
                nn.ReLU(out_type, inplace=True)
            )
            e2cnn_layers.append(block_i)
            hidden_image_size.append({"in": hidden_image_size[-1]["out"], "out": hidden_image_size[-1]["out"]})
            
            if i % 2 == 0 and hidden_image_size[-1]["out"] % 2 == 1:
                pool_i = nn.SequentialModule(
                    nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2)
                )
                e2cnn_layers.append(pool_i)
                hidden_image_size.append({"in": hidden_image_size[-1]["out"], "out": int(hidden_image_size[-1]["out"]/2)+1})
            
            elif i % 2 == 0 and hidden_image_size[-1]["out"] % 2 == 0:
                pool_i = nn.SequentialModule(
                    nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=1)
                )
                e2cnn_layers.append(pool_i)
                hidden_image_size.append({"in": hidden_image_size[-1]["out"], "out": hidden_image_size[-1]["out"]})
        
        # Define C_N invariant layer
        gpool = nn.GroupPooling(e2cnn_layers[-1].out_type)
        e2cnn_layers.append(gpool)
        hidden_image_size.append({"in": hidden_image_size[-1]["out"], "out": hidden_image_size[-1]["out"]})
        
        # Average pixel information
        avgpool = nn.PointwiseAvgPool(e2cnn_layers[-1].out_type, hidden_image_size[-1]["out"])
        e2cnn_layers.append(avgpool)
        hidden_image_size.append({"in": hidden_image_size[-1]["out"], "out": 1})
        
        # Wrap everything in nn.SequentialModule
        self.invariant_layer = nn.SequentialModule(*e2cnn_layers)
        
        # number of output channels
        c = self.invariant_layer.out_type.size
        
        # Fully Connected
        mlp_layers.append(torch.nn.Linear(c, hidden_dim))
        mlp_layers.append(torch.nn.BatchNorm1d(hidden_dim))
        mlp_layers.append(torch.nn.ELU(inplace=True))
        
        # MLP layers
        for _ in range(num_hidden_layers - 1):
            mlp_layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
            mlp_layers.append(torch.nn.ReLU())
        
        # Final layer
        mlp_layers.append(torch.nn.Linear(hidden_dim, output_dim))
        
        self.mlp = torch.nn.Sequential(*mlp_layers)
    
    def forward(self, input):
        # wrap the input tensor in a GeometricTensor
        x = nn.GeometricTensor(input, self.input_type)
        
        latent = self.invariant_layer(x)
        latent = latent.tensor
        
        # classify with the final fully connected layers
        output = self.mlp(latent.reshape(latent.shape[0], -1))
        
        return output
    
    @classmethod
    def load_from_checkpoint(cls, filepath, optimizer_class=torch.optim.Adam):
        """
        Class method to load config, model, and optimizer from filepath.

        Returns:
            model (CNSteerableCNN): Restored model
            optimizer (torch.optim.Optimizer): Restored optimizer
            checkpoint (dict): Additional checkpoint information (epoch, loss, etc.)
        """
        checkpoint = torch.load(filepath, map_location='cpu')

        config = checkpoint['config']
        model = cls(**config)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)

        optimizer = optimizer_class(model.parameters())
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        return model, optimizer, checkpoint

    