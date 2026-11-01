from .loader import speed_extension


def kernel_extension(config, forward):
    wide = config[0 if forward else 1]
    return speed_extension(wide, wide, config[2] if wide else 2)
