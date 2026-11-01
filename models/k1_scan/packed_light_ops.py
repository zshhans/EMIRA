DEFAULT_CONFIG = (1, 1, 2, 1)


def layout(length, wide):
    if length <= 128:
        return 32, 4
    if length <= 256:
        return 32, 8
    if length <= 512:
        return 32, 16
    if length <= 1024:
        return (128, 8) if wide else (64, 16)
    return (256, 8) if wide else (128, 16)
