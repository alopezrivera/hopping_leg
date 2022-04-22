import numpy as np


def csv(params, motors, name):

    n = len(params[0])

    data = np.empty((n, len(motors)*len(params)+1))
    header = ''
    i = 0
    for motor in motors:
        for param in params:
            data[:, i] = param
            header += (',' if i > 0 else '') + param.replace('_', ' ').title()
            i += 1
    np.savetxt(name, data.T, delimiter=',', header=header, comments='')