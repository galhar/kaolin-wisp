# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import torch.nn as nn
from wisp.core import RenderBuffer

from wisp.models.decoders import DecoderFeatureMapToImage
from wisp.models.nefs import BaseNeuralField
from wisp.tracers.base_tracer import BaseTracer


class Pipeline(nn.Module):
    """Base class for implementing neural field pipelines.

    Pipelines consist of several components:

        - Neural fields (``self.nef``) which take coordinates as input and outputs signals.
          These usually consist of several optional components:

            - A feature grid (``self.nef.grid``)
              Sometimes also known as 'hybrid representations'.
            - An acceleration structure (``self.nef.grid.blas``) which can be used to accelerate spatial queries.
            - A decoder (``self.net.decoder``) which can take the features (or coordinates, or embeddings) and covert it to signals.

        - A forward map (``self.tracer``) which is a function which will invoke the pipeline in
          some outer loop. Usually this consists of renderers which will output a RenderBuffer object.
    
    The 'Pipeline' classes are responsible for holding and orchestrating these components.
    """
    
    def __init__(self, nef: BaseNeuralField, tracer: BaseTracer = None):
        """Initialize the Pipeline.

        Args:
            nef (nn.Module): Neural fields module.
            tracer (nn.Module or None): Forward map module.
        """
        super().__init__()
    
        self.nef: BaseNeuralField = nef
        self.tracer: BaseTracer = tracer

    def forward(self, *args, **kwargs):
        """The forward function will use the tracer (the forward model) if one is available. 
        
        Otherwise, it'll execute the neural field.
        """
        if self.tracer is not None:
            return self.tracer(self.nef, *args, **kwargs)
        else:
            return self.nef(*args, **kwargs)


class Pipeline2DDecoder(nn.Module):
    """Base class for implementing neural field pipelines.

    Pipelines consist of several components:

        - Neural fields (``self.nef``) which take coordinates as input and outputs signals.
          These usually consist of several optional components:

            - A feature grid (``self.nef.grid``)
              Sometimes also known as 'hybrid representations'.
            - An acceleration structure (``self.nef.grid.blas``) which can be used to accelerate spatial queries.
            - A decoder (``self.net.decoder``) which can take the features (or coordinates, or embeddings) and covert it to signals.

        - A forward map (``self.tracer``) which is a function which will invoke the pipeline in
          some outer loop. Usually this consists of renderers which will output a RenderBuffer object.

    The 'Pipeline' classes are responsible for holding and orchestrating these components.
    """

    def __init__(self, nef: BaseNeuralField, decoder2D: DecoderFeatureMapToImage, tracer: BaseTracer = None):
        """Initialize the Pipeline.

        Args:
            nef (nn.Module): Neural fields module.
            tracer (nn.Module or None): Forward map module.
        """
        super().__init__()

        self.nef: BaseNeuralField = nef
        self.decoder2D: DecoderFeatureMapToImage = decoder2D
        self.tracer: BaseTracer = tracer

    def forward(self, *args, **kwargs):
        """The forward function will use the tracer (the forward model) if one is available.

        Otherwise, it'll execute the neural field.
        """
        if self.tracer is not None:
            rb = self.tracer(self.nef, *args, **kwargs)
            ret_buffer_dict = {k:v for k,v in rb}
            grid_features = rb.grid_features
            rgb = self.decoder2D(grid_features)
            ret_buffer_dict['rgb'] = rgb
            ret_buffer_dict['grid_features'] = grid_features
            return RenderBuffer(**ret_buffer_dict)
        else:
            return self.nef(*args, **kwargs)

