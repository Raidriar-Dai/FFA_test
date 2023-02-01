# For x: 0_th dimension：Sample Size; 1_st dimension：Flattened Features
# y is a scaler: 0 <= y <= 9
# Cover each row of x (each sample in x) with the same label,
# by writing x.max() into the position which corresponds to the label value.
def overlay_y_on_x(x, y):
    x_ = x.clone()
    x_[:, :10] *= 0.0   # REPLACE the first 10 pixels by the label representation.
    x_[range(x.shape[0]), y] = x.max()
    return x_