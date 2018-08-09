# -*- coding: utf-8 -*-

'''Import MatchPool and corresponding gradient operations.'''


from tensorflow.python.framework import ops
import tensorflow as tf
import os


# GPU version
root = os.path.dirname(os.path.abspath(__file__))
_match_pool_gpu_module = tf.load_op_library(
    os.path.join(root, "match_pool_gpu.so"))
_match_pool_grad_gpu_module = tf.load_op_library(
    os.path.join(root, "match_pool_grad_gpu.so"))
match_pool = _match_pool_gpu_module.match_pool
match_pool_grad = _match_pool_grad_gpu_module.match_pool_grad


@ops.RegisterGradient("MatchPool")
def _match_pool_grad(op, output_grad, mask_grad, distances_grad):
    '''Gradient of MatchPool operation. If the `op` has N outputs,
    then `grad(s)` should be N inputs, just like above. What is
    returned is a list with length of input list of this `op`, and
    stores the gradients w.r.t. each input.

    This function will be called automatically when using `tf.gradients`
    in `tf.compute_gradients`, refer to `match_pool_grad_test.py` in this 
    folder for more infos.

    param: op: 
        MatchPool op, defined by `ops.RegisterGradient("MatchPool")`.
    param: output_grad:
        gradient of `output` got MatchPool op.
    param: mask_grad:
        gradient of `mask` got by MatchPool op.
    param: distances_grad:
        gradient of `distances` got by MatchPool op.
    return: input_grad:
        gradient of (the only single) `input` of MatchPool op.
    '''
    input_tensor = op.inputs[0]
    mask = op.outputs[1]
    topk = op.get_attr('topk')
    input_grad = match_pool_grad(input=input_tensor, 
        grad=output_grad, mask=mask, topk=topk)
    return [input_grad]