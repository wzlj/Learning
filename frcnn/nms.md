## source repo


https://github.com/endernewton/tf-faster-rcnn

## source code

tf-faster-rcnn/lib/layer_utils/proposal_layer.py

  keep = nms(np.hstack((proposals, scores)), nms_thresh)  # Non-maximal suppression
