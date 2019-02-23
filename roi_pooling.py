
def roi_pooling_forward_cpu(input_data, input_rois, spatial_scale, pool_size):

    N, C, H, W = input_data.shape
    pooled_width = pooled_height = pool_size

    int_input_rois = np.zeros_like(input_rois, dtype=np.int32)
    int_input_rois[:, 0] = input_rois[:, 0].astype(np.int32)
    int_input_rois[:, 1:] = np.round(input_rois[:, 1:] * spatial_scale).astype(np.int32)

    m_shape = (input_rois.shape[0], C, pooled_height, pooled_width)
    # out_put = np.random.normal(0, 0.1, size=m_shape).astype(np.float32)
    out_put = np.zeros(m_shape, dtype=np.float32)
    print("\n______________________11111_________________________\n")
    for i in range(int_input_rois.shape[0]):

        batch_id, roi_start_w, roi_start_h, roi_end_w, roi_end_h = int_input_rois[i]
        # roi_start_w = int(round(roi_start_w * spatial_scale))
        # roi_end_w = int(round(roi_end_w * spatial_scale))
        # roi_start_h = int(round(roi_start_h * spatial_scale))
        # roi_end_h = int(round(roi_end_h * spatial_scale))

        roi_width = max(roi_end_w - roi_start_w + 1, 1)
        roi_height = max(roi_end_h - roi_start_h + 1, 1)
        bin_size_h = float(roi_height) / float(pooled_height)
        bin_size_w = float(roi_width) / float(pooled_width)
        print("roi_start_w, roi_end_w, roi_start_h, roi_end_h", roi_start_w, roi_end_w, roi_start_h, roi_end_h)

        for ph in range(pooled_height):
            for pw in range(pooled_width):
                # index =
                h_start = int(np.floor(ph * bin_size_h))
                h_end = int(np.ceil((ph + 1) * bin_size_h))
                w_start = int(np.floor(pw * bin_size_w))
                w_end = int(np.ceil((pw + 1) * bin_size_w))

                # check boundary
                h_start = min(max(h_start + roi_start_h, 0), H)
                h_end = min(max(h_end + roi_start_h, 0), H)
                w_start = min(max(w_start + roi_start_w, 0), W)
                w_end = min(max(w_end + roi_start_w, 0), W)

                # is_empty = (h_end <= h_start) or (w_end <= w_start)
                # pool_index = h * pooled_width + w
                if h_end <= h_start or w_end <= w_start:
                    continue
                for j in range(C):
                    out_put[i, j, ph, pw] = np.max(input_data[int(batch_id), j, h_start: h_end, w_start: w_end])

    return out_put
