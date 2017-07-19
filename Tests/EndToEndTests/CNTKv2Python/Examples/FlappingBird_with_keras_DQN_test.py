# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import numpy as np
import os
import sys

os.environ["SDL_VIDEODRIVER"] = "dummy" 

def set_keras_backend(backend):
    from keras import backend as K
    if K.backend() != backend:
        os.environ['KERAS_BACKEND'] = backend
        reload(K)
        assert K.backend() == backend


from cntk.device import try_set_default_device, gpu

abs_path = os.path.dirname(os.path.abspath(__file__))
example_dir = os.path.join(abs_path, "..", "..", "..", "..",
                             "Examples", "ReinforcementLearning", 
                             "FlappingBirdWithKeras")
sys.path.append(example_dir)
current_dir = os.getcwd()
os.chdir(example_dir)

# This example shows how the same keras model that runs on 
# Tensorflow as posted at the github site of the author Ben Lau 
# https://github.com/yanpanlau/Keras-FlappyBird) can be run
# with CNTK API with minimum code change.
# Note: please ignore the UserWarning the test emits due to a newer 
# API for Convolution2D function available.
 
def test_FlappingBird_with_keras_DQN_noerror(device_id):
    from cntk.ops.tests.ops_test_utils import cntk_device
    try_set_default_device(cntk_device(device_id))
    
    set_keras_backend("cntk")
    
    sys.path.append(example_dir)
    current_dir = os.getcwd()
    os.chdir(example_dir)
    
    import FlappingBird_with_keras_DQN as fbgame

    model = fbgame.buildmodel()
    args = {'mode': 'Run'}
    res = fbgame.trainNetwork(model, args, internal_testing=True )
    
    np.testing.assert_array_equal(res, 0, err_msg='Error in running Flapping Bird example', verbose=True)
    
    args = {'mode': 'Train'}
    res = fbgame.trainNetwork(model, args, internal_testing=True )
    
    np.testing.assert_array_equal(res, 0, err_msg='Error in testing Flapping Bird example', verbose=True)
    
    os.chdir(current_dir)
    print("Done")




