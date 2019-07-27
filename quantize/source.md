https://tensorflow.google.cn/lite/convert/python_api

https://segmentfault.com/a/1190000018741468

https://zhuanlan.zhihu.com/p/42811261

https://blog.csdn.net/u011092156/article/details/80642133

https://www.jianshu.com/p/19467624b4b0

'''

inline int32 MultiplyByQuantizedMultiplierSmallerThanOne(
    int32 x, int32 quantized_multiplier, int right_shift) {
  using gemmlowp::RoundingDivideByPOT;
  using gemmlowp::SaturatingRoundingDoublingHighMul;
  return RoundingDivideByPOT(
      SaturatingRoundingDoublingHighMul(x, quantized_multiplier), right_shift);
}

template <>
inline std::int32_t SaturatingRoundingDoublingHighMul(std::int32_t a,
                                                      std::int32_t b) {
  bool overflow = a == b && a == std::numeric_limits<std::int32_t>::min();
  std::int64_t a_64(a);
  std::int64_t b_64(b);
  std::int64_t ab_64 = a_64 * b_64;
  std::int32_t nudge = ab_64 >= 0 ? (1 << 30) : (1 - (1 << 30));
  std::int32_t ab_x2_high32 =
      static_cast<std::int32_t>((ab_64 + nudge) / (1ll << 31));
  return overflow ? std::numeric_limits<std::int32_t>::max() : ab_x2_high32;
}

// Correctly-rounded-to-nearest division by a power-of-two.
// Also known as a rounding arithmetic right shift.
template <typename IntegerType, typename ExponentType>
inline IntegerType RoundingDivideByPOT(IntegerType x, ExponentType exponent) {
  assert(exponent >= 0);
  assert(exponent <= 31);
  const IntegerType mask = Dup<IntegerType>((1ll << exponent) - 1);
  const IntegerType zero = Dup<IntegerType>(0);
  const IntegerType one = Dup<IntegerType>(1);
  const IntegerType remainder = BitAnd(x, mask);
  const IntegerType threshold =
      Add(ShiftRight(mask, 1), BitAnd(MaskIfLessThan(x, zero), one));
  return Add(ShiftRight(x, exponent),
             BitAnd(MaskIfGreaterThan(remainder, threshold), one));
}


inline void Conv(const uint8* input_data, const Dims<4>& input_dims,
                 int32 input_offset, const uint8* filter_data,
                 const Dims<4>& filter_dims, int32 filter_offset,
                 const int32* bias_data, const Dims<4>& bias_dims,
                 int stride_width, int stride_height, int pad_width,
                 int pad_height, int32 output_offset, int32 output_multiplier,
                 int output_shift, int32 output_activation_min,
                 int32 output_activation_max, uint8* output_data,
                 const Dims<4>& output_dims, uint8* im2col_data,
                 const Dims<4>& im2col_dims,
                 gemmlowp::GemmContext* gemm_context) {
  (void)im2col_data;   // only used in optimized code.
  (void)im2col_dims;   // only used in optimized code.
  (void)gemm_context;  // only used in optimized code.
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  const int batches = MatchingArraySize(input_dims, 3, output_dims, 3);
  const int input_depth = MatchingArraySize(input_dims, 0, filter_dims, 0);
  const int output_depth =
      MatchingArraySize(filter_dims, 3, bias_dims, 0, output_dims, 0);
  const int input_height = ArraySize(input_dims, 2);
  const int input_width = ArraySize(input_dims, 1);
  const int filter_height = ArraySize(filter_dims, 2);
  const int filter_width = ArraySize(filter_dims, 1);
  const int output_height = ArraySize(output_dims, 2);
  const int output_width = ArraySize(output_dims, 1);
  for (int batch = 0; batch < batches; ++batch) {
    for (int out_y = 0; out_y < output_height; ++out_y) {
      for (int out_x = 0; out_x < output_width; ++out_x) {
        for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
          const int in_x_origin = (out_x * stride_width) - pad_width;
          const int in_y_origin = (out_y * stride_height) - pad_height;
          int32 acc = 0;
          for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
            for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
              for (int in_channel = 0; in_channel < input_depth; ++in_channel) {
                const int in_x = in_x_origin + filter_x;
                const int in_y = in_y_origin + filter_y;
                // If the location is outside the bounds of the input image,
                // use zero as a default value.
                if ((in_x >= 0) && (in_x < input_width) && (in_y >= 0) &&
                    (in_y < input_height)) {
                  int32 input_val = input_data[Offset(input_dims, in_channel,
                                                      in_x, in_y, batch)];
                  int32 filter_val =
                      filter_data[Offset(filter_dims, in_channel, filter_x,
                                         filter_y, out_channel)];
                  acc +=
                      (filter_val + filter_offset) * (input_val + input_offset);
                }
              }
            }
          }
          if (bias_data) {
            acc += bias_data[Offset(bias_dims, out_channel, 0, 0, 0)];
          }
          acc = MultiplyByQuantizedMultiplierSmallerThanOne(
              acc, output_multiplier, output_shift);
          acc += output_offset;
          acc = std::max(acc, output_activation_min);
          acc = std::min(acc, output_activation_max);
          output_data[Offset(output_dims, out_channel, out_x, out_y, batch)] =
              static_cast<uint8>(acc);
        }
      }
    }
  }
}

'''
