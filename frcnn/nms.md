## source repo


https://github.com/endernewton/tf-faster-rcnn

## source code

tf-faster-rcnn/lib/layer_utils/proposal_layer.py

    keep = nms(np.hstack((proposals, scores)), nms_thresh)  # Non-maximal suppression
  
    indices = tf.image.non_max_suppression(proposals, scores, max_output_size=post_nms_topN, iou_threshold=nms_thresh)
