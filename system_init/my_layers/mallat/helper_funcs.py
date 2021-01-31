__all__ =["calculate_channels_after_scat", "find_best_J", "check_shape"]

def calculate_channels_after_scat(in_channels, J, order, L=8):
    order0_size = 1
    order1_size = L * J
    order2_size = L ** 2 * J * (J - 1) // 2
    output_size = order0_size + order1_size
    if (order == 2):
        output_size+=order2_size
    return int(in_channels*output_size)

#If J is not specified, this functions will find the biggest J that can used with given input params
def find_best_J(shape, order, J=-1):
    check_shape(shape)
    if J is not -1:
        return J
    else:
        j = 1
        while j > 0:
            out_shape = shape[1]
            for order in range(order):
                out_shape *= 2 ** (-j)
            if out_shape % 1 == 0:
                j += 1
            else:
                j -= 1
                break
        return j

def check_shape(shape):
    if len(shape) is not 3:
        raise ValueError("The shape should be of the form CxHxW where C - number of input channels, W anf H are width and hight respectively.")
    if shape[1] is not shape[2]:
        raise ValueError("H and W should be equal in the shape variable.")

    return shape