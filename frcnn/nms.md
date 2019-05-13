## source repo


https://github.com/endernewton/tf-faster-rcnn

## source code

tf-faster-rcnn/lib/layer_utils/proposal_layer.py

    1. keep = nms(np.hstack((proposals, scores)), nms_thresh)  # Non-maximal suppression
  
    2. indices = tf.image.non_max_suppression(proposals, scores, max_output_size=post_nms_topN, iou_threshold=nms_thresh)  ==>
    
        gen_image_ops.non_max_suppression_v3(boxes, scores, max_output_size, iou_threshold, score_threshold)
        
        non_max_suppression_v3(boxes, scores, max_output_size, iou_threshold, score_threshold, name=None):  ==>
        
        non_max_suppression_v3_eager_fallback( boxes, scores, max_output_size, iou_threshold, 
        score_threshold, name=name, ctx=_ctx)  ==>
        
        _result = _execute.execute(b"NonMaxSuppressionV3", 1, inputs=_inputs_flat, attrs=_attrs, ctx=_ctx, name=name) ==> 
