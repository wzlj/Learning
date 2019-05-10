## source repo


https://github.com/endernewton/tf-faster-rcnn

## source code

tf-faster-rcnn/lib/layer_utils/proposal_layer.py

  # Non-maximal suppression
  keep = nms(np.hstack((proposals, scores)), nms_thresh)
